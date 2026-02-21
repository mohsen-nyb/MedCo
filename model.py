

import torch
from torch import nn
from pyhealth.models import BaseModel, TransformerLayer
from pyhealth.tokenizer import Tokenizer
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, RGATConv, RGCNConv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import login
import random
from tqdm import tqdm


login("hf_...")

class CoMed(BaseModel):
    def __init__(self,
                 dataset: list,
                 feature_keys: list,
                 label_key: str,
                 mode: str,
                 #general EHR hyperparams
                 embedding_dim=128,
                 dropout=0.5,
                 nheads=1,
                 nlayers=1,
                 #globe GNN hyperparams
                 nlayers_gnn=2,
                 n_gat_heads=1,
                 gnn_dropout=0.4,
                 # ---- LLM integration options ----
                 use_llm_for_node: bool = True,
                 use_freezed_llm_for_node: bool = True,
                 no_random_emb: bool = False,
                 init_w_freeze_llm: bool = False,
                 use_edge_attr: bool = True,
                 llm_node_path: str = None,
                 llm_edge_path: str = None,
                 llm_residual_scale: float = 0.01,
                 edge_attr_method='stat',
                 device='cuda',
                 #lamma+lora params
                 llm_name="meta-llama/Llama-3.2-1B",
                 lora_r=8,
                 lora_alpha=32,
                 max_updates_per_feat=10,
                 max_epoch_to_train_node_llm=150,
                 min_epoch_to_train_node_llm=-1,
                 save_last_lora: bool = False,
                 load_last_lora: bool = False,
                 lora_save_dir: str = None,
                 lora_load_dir: str = None,
                 seed=None,
                 **kwargs):
        super().__init__(dataset, feature_keys, label_key, mode)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)

        self.dataset = dataset
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.embedding_dim = embedding_dim

        # LLM options
        self.use_llm_for_node = use_llm_for_node
        self.max_epoch_to_train_node_llm=max_epoch_to_train_node_llm
        self.min_epoch_to_train_node_llm=min_epoch_to_train_node_llm
        if use_llm_for_node:
            self.use_freezed_llm_for_node= use_freezed_llm_for_node
            self.init_w_freeze_llm = init_w_freeze_llm
            self.no_random_emb=no_random_emb
        self.llm_edge_path = llm_edge_path
        self.llm_node_path = llm_node_path
        self.llm_residual_scale = llm_residual_scale
        self.use_edge_attr=use_edge_attr

        self.feat_tokenizers3 = {}
        self.embeddings3 = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()

        # create feature transform layers (tokenizers + embedding modules) first
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            self._my_add_feature_transform_layer(
                feature_key, input_info, special_tokens=["<pad>", "<unk>"]
            )

        # transformer per feature (unchanged)
        self.transformer = nn.ModuleDict()
        for feature_key in feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim, heads=nheads, dropout=dropout, num_layers=nlayers, **kwargs
            )

        output_size = self.get_output_size(self.get_label_tokenizer())
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

        # graph edges + edge MLP (unchanged)
        self.edge_attr_method = edge_attr_method
        self.num_relations=None
        self.edge_index, self.edge_attr_stat, self.edge_rel_type, self.edge_attr_reasoning = self.build_KG_edges()
        # after you load raw edge feature dims in __init__ (edge_feat_dim, edge_reason_dim)

        if use_edge_attr:
            self.edge_feat_proj = nn.Sequential(nn.Linear(self.edge_attr_stat.shape[-1], self.embedding_dim),
                                                nn.LayerNorm(self.embedding_dim), nn.ReLU())
            # self.edge_reason_proj = nn.Sequential(nn.Linear(self.edge_attr_reasoning.shape[-1], self.embedding_dim),
            #                                       nn.LayerNorm(self.embedding_dim), nn.ReLU())
            if edge_attr_method == 'concat':
                edge_dim=2*self.embedding_dim
            else:
                edge_dim = self.embedding_dim
        else:
            edge_dim=None


        self.gnn_layers = nn.ModuleList()
        num_gnn_layers = max(1, nlayers_gnn)
        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                GATConv(embedding_dim, embedding_dim, heads=n_gat_heads, concat=False, dropout=gnn_dropout, edge_dim=edge_dim, add_self_loops=True)
            )
        self.gnn_layernorm = nn.LayerNorm(embedding_dim)
        self.gnn_dropout = nn.Dropout(gnn_dropout)
        self.gnn_alpha = nn.Parameter(torch.tensor(0.5))

        # Initialize embedding weights small (they will act as residual on top of projected LLM)
        for k in self.embeddings3:
            nn.init.normal_(self.embeddings3[k].weight, mean=0.0, std=self.llm_residual_scale)

        self.save_last_lora = save_last_lora
        self.load_last_lora = load_last_lora
        self.lora_save_dir = lora_save_dir
        self.lora_load_dir = lora_load_dir
        self.max_updates_per_feat=max_updates_per_feat
        if self.use_llm_for_node:
            self.init_llm_for_node(llm_name=llm_name, r=lora_r, lora_alpha=lora_alpha)



    def _my_add_feature_transform_layer(self, feature_key: str, info, special_tokens=None):
        tokens3 = self.dataset.get_all_tokens(feature_key)

        if info["type"] == str:
            if special_tokens is None:
                special_tokens = ["<pad>", "<unk>"]
            tokenizer3 = Tokenizer(
                tokens=tokens3,
                special_tokens=special_tokens,
            )
            self.feat_tokenizers3[feature_key] = tokenizer3

             
            emb = nn.Embedding(
                tokenizer3.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer3.get_padding_index(),
            )
            self.embeddings3[feature_key] = emb

    def init_llm_for_node(self, llm_name, r, lora_alpha):
 
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            # device_map="auto",            # optional: automatic device placement
            # load_in_8bit=True,           # optional: needs bitsandbytes
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.llm_hidden_size = self.llm_model.config.hidden_size
        # load tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=False)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            try:
                self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
            except Exception:
                pass
        self.lamma_node_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.embedding_dim),
            nn.Dropout(0.2),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU()
        )

        # LoRA config - adjust r, lora_alpha, target_modules as needed for your model
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj","v_proj","k_proj","o_proj","down_proj","up_proj"],  # common for Llama-like models; tweak if necessary
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
 
        if getattr(self, "load_last_lora", False) and os.path.isdir(self.lora_load_dir):
            # create PeftModel using stored adapter weights
            try:
                # PeftModel.from_pretrained will attach the PEFT adapter to the base model
                self.llm_model = PeftModel.from_pretrained(self.llm_model, self.lora_load_dir)
                print(f"Loaded LoRA weights from {self.lora_load_dir} for nodes")
            except Exception as e:
                print(f"Failed to load LoRA from {self.lora_load_dir} for nodes, falling back to new LoRA. Error: {e}")
                self.llm_model = get_peft_model(self.llm_model, lora_config)
        else:
            print("create fresh LoRA-injected model")
            self.llm_model = get_peft_model(self.llm_model, lora_config)



        for n, p in self.llm_model.named_parameters():
            if "lora_" in n or "adapter" in n or "peft" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for p in self.lamma_node_proj.parameters():
            p.requires_grad = True

        self.llm_hidden_size = self.llm_model.config.hidden_size

        self._prapre_node_text_prompts_for_llm_and_cash_embd()  # keep node prompt texts

        self.dx_llm_alpha = nn.Parameter(torch.tensor(0.5))
        self.rx_llm_alpha = nn.Parameter(torch.tensor(0.5))
        self.px_llm_alpha = nn.Parameter(torch.tensor(0.5))


    def _prapre_node_text_prompts_for_llm_and_cash_embd(self):
        path = self.llm_node_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"LLM embedding file not found at {path}")

        df_node_llm_embd = pd.read_parquet(path)
        #prompts = list(df_node_llm_embd['prompt_text'].values) #(1262, )

        # build mapping: key = (type, org_code) -> raw vector
        llm_map = {}
        for _, row in df_node_llm_embd.iterrows():
            code = row['org_code']
            if row['type']=='dx':
                t='conditions'
            elif row['type']=='rx':
                t ='drugs'
            elif row['type']=='px':
                t = 'procedures'
            llm_map[(t, str(code))] = str(row['prompt_text'])  # ensure same type as tokens (string)

        # For each feature (conditions, drugs, procedures) create raw buffer aligned with tokenizer order
        for feature_key in self.feature_keys:
            tokens = self.dataset.get_all_tokens(feature_key)
            token_indices = list(self.feat_tokenizers3[feature_key].convert_tokens_to_indices(tokens))
            node_text_lst = ['None']*self.embeddings3[feature_key].weight.shape[0]
            # tokens should match tokenizer vocab order used earlier
            for i in range(len(tokens)):
                # depending on how your org_code is recorded, you might need to map 'tok' to code key
                # here we try a few alternatives:
                found = llm_map.get((feature_key, tokens[i]), None)
                if found is None:
                    print('not found')
                    if token_indices[i] not in [0,1]:
                        node_text_lst[token_indices[i]] = f"feature_key code {str(tokens[i])}"
                else:
                    node_text_lst[token_indices[i]]  = found
            # register raw tensor as buffer (moved to device in forward)
            buf_name = f"prompt_text_{feature_key}"
            setattr(self, buf_name, node_text_lst)

            # minimal batched encode + save cache (replace your old block)
            cache_name = f"llm_proj_cache_{feature_key}"
            num_codes = self.embeddings3[feature_key].weight.shape[0]
            # disk cache path (same folder as node parquet)
            default_cache_file  = os.path.join(f"../saved_files/mimic4/KG_openai/nodes/raw_llamma1B_cache/_{feature_key}_raw_cache.pt")
            lora_cache_file = os.path.join(self.lora_load_dir, f"llm_proj_cache_{feature_key}.pt")
            if getattr(self, "load_last_lora", False) and os.path.exists(lora_cache_file):
                print(f'loading latest saved lora_cache: {lora_cache_file}')
                cache_file = lora_cache_file
            else:
                print(f'loading defualt raw lora_cache: {default_cache_file}')
                cache_file = default_cache_file
            # If disk cache exists, load and register it
            if os.path.exists(cache_file):
                raw_cache = torch.load(cache_file, map_location=self.embeddings3[feature_key].weight.device).float()
                if raw_cache.shape[0] != num_codes:
                    raise RuntimeError(f"Cached raw LLM matrix rows {raw_cache.shape[0]} != expected {num_codes}")
                self.register_buffer(cache_name, raw_cache)
                print(f'cache_file loaded')
            else:
                llm_batch_size = 100
                encoded_rows = []
                texts = node_text_lst[2:]
                self.llm_model.eval()
                for start in tqdm(range(0, len(texts), llm_batch_size)):
                    batch_texts = texts[start:start + llm_batch_size]
                    with torch.no_grad():
                        batch_emb = self.encode_texts_with_prompt(batch_texts).to(self.embeddings3[feature_key].weight.device)
                    encoded_rows.append(batch_emb)
                    lamma_out = torch.cat(encoded_rows, dim=0)
                assert num_codes == 2+lamma_out.shape[0]
                lamma_init_cash_emb = torch.cat([torch.zeros(2, lamma_out.shape[-1]), lamma_out], dim=0)
                torch.save(lamma_init_cash_emb, cache_file)
                assert num_codes == lamma_init_cash_emb.shape[0]
                self.register_buffer(cache_name, lamma_init_cash_emb)
                self.llm_model.train()

            if self.init_w_freeze_llm:
                self.embeddings3[feature_key] = nn.Embedding.from_pretrained(getattr(self, cache_name), freeze=False)


    def save_lora_and_caches(self, lora_save_dir: str = None):
        """
        Save PEFT (LoRA) adapter weights and the per-feature llm cache buffers to disk.
        Call this after training to persist LoRA and the final cache embeddings.
        """
        if lora_save_dir is None:
            lora_save_dir = self.lora_save_dir

        os.makedirs(lora_save_dir, exist_ok=True)

        # Save PEFT/LoRA adapter
        try:
            # PeftModel / get_peft_model has save_pretrained
            self.llm_model.save_pretrained(lora_save_dir)
            print(f"Saved LoRA/PEFT adapter to {lora_save_dir}")
        except Exception as e:
            print(f"Warning: failed to save LoRA model: {e}")

        # Save each llm_proj_cache_<feature> buffer to the same lora_save_dir for convenience
        for feature_key in self.feature_keys:
            cache_name = f"llm_proj_cache_{feature_key}"
            if hasattr(self, cache_name):
                cache_tensor = getattr(self, cache_name)
                # Compose filename (use original cache pattern if you want)
                cache_file = os.path.join(lora_save_dir, f"llm_proj_cache_{feature_key}.pt")
                try:
                    # Move to cpu first to ensure compatibility; keep float32 for safety
                    torch.save(cache_tensor.detach().cpu(), cache_file)
                    print(f"Saved cache for {feature_key} -> {cache_file}")
                except Exception as e:
                    print(f"Warning: failed to save cache for {feature_key}: {e}")
            else:
                print(f"No buffer {cache_name} to save.")



    def make_undirected_with_metadata(self, edge_index, edge_attr=None, edge_rel_type=None):
        #edge_index: [2, E], edge_attr: [E, F], edge_rel_type: [E] (long)

        # reverse edges
        rev = edge_index[[1, 0], :]
        edge_index_bi = torch.cat([edge_index, rev], dim=1)  # [2, 2E]
        if edge_attr is not None:
            edge_attr_bi = torch.cat([edge_attr, edge_attr], dim=0)  # [2E, F]
        else:
            edge_attr_bi = None
        if edge_rel_type is not None:
            edge_rel_type_bi = torch.cat([edge_rel_type, edge_rel_type], dim=0)  # [2E]
        else:
            edge_rel_type_bi = None
        return edge_index_bi, edge_attr_bi, edge_rel_type_bi

    def build_KG_edges(self):
        print('start build_KG_edges ...')
        df = pd.read_csv('../saved_files/mimic4/KG_openai/batch/gpt5_mini/stitched_edges_fixed_edge_weight.csv')
        edge_feat_matrix = np.load('../saved_files/mimic4/KG_openai/batch/gpt5_mini/edge_features.npy')
        #edge_reasoning_embeddings = np.load(self.llm_edge_path)
        df['edge_feature'] = list(edge_feat_matrix.astype(np.float32))
        #df['edge_reasoning_embeddings'] = list(edge_reasoning_embeddings.astype(np.float32))
        df = df[~ df['predicted_relationship'].isin(['cannot_decide', 'no_significant_relation'])][
            ['code1_id', 'code2_id', 'type1', 'type2', 'edge_feature', 'predicted_relationship']]
        df = df.drop_duplicates(subset=['code1_id', 'code2_id', 'type1', 'type2'])


        node_idx_dict = {}

        codes_dx = self.dataset.get_all_tokens('conditions')
        codes_dx_indices = list(self.feat_tokenizers3['conditions'].convert_tokens_to_indices(codes_dx))
        node_idx_dict['dx'] = dict(zip(codes_dx, codes_dx_indices))

        codes_rx = self.dataset.get_all_tokens('drugs')
        codes_rx_indices = list(self.feat_tokenizers3['drugs'].convert_tokens_to_indices(codes_rx))
        codes_rx_indices = list(np.array(codes_rx_indices) + self.embeddings3['conditions'].weight.shape[0])
        node_idx_dict['rx'] = dict(zip(codes_rx, codes_rx_indices))

        codes_px = self.dataset.get_all_tokens('procedures')
        codes_px_indices = list(self.feat_tokenizers3['procedures'].convert_tokens_to_indices(codes_px))
        codes_px_indices = list(np.array(codes_px_indices) + self.embeddings3['conditions'].weight.shape[0] + self.embeddings3['drugs'].weight.shape[0])
        node_idx_dict['px'] = dict(zip(codes_px, codes_px_indices))

        def row_to_edge_tuple_idxs(row):
            mapping_dict1 = node_idx_dict[row['type1']]
            mapping_dict2 = node_idx_dict[row['type2']]
            return pd.Series([mapping_dict1.get(row['code1_id']), mapping_dict2.get(row['code2_id'])])

        df[['edge_idx_code_1', 'edge_idx_code_2']] = df.apply(row_to_edge_tuple_idxs, axis=1)
        edge_index = torch.tensor(df[['edge_idx_code_1', 'edge_idx_code_2']].values, dtype=torch.long).T
        edge_attr = torch.tensor(np.stack(df['edge_feature'].values), dtype=torch.float)
        #edge_attr_reasoning = torch.tensor(np.stack(df['edge_reasoning_embeddings'].values), dtype=torch.float)
        df['predicted_relationship_id'] = pd.factorize(df['predicted_relationship'])[0]
        edge_rel_type = torch.tensor(df['predicted_relationship_id'].values, dtype=torch.long)
        self.num_relations = df['predicted_relationship_id'].nunique()

        print('done build_KG_edges ...')


        return edge_index, edge_attr, edge_rel_type, None



    def combine_edge_features(
            self,
            edge_attr: torch.Tensor,
            method: str = 'concat',
            proj_layer_stat: nn.Module = None,
    ) -> torch.Tensor:
        """
        Combine and optionally project raw edge features.

        Args:
            edge_attr: Tensor [E, F_stat] or None
            edge_attr_reasoning: Tensor [E, F_reason] or None
            method: one of {'stat', 'reasoning', 'concat', 'weighted_sum', 'projected_concat'}
            reasoning_weight: used when method == 'weighted_sum'
            proj_layer_stat: optional nn.Module to project stat features -> common dim
            proj_layer_reasoning: optional nn.Module to project reasoning features -> common dim

        Returns:
            Tensor [E, D] combined edge features ready for GNN (on device of projections / model).
        """
        # Basic validation
        if edge_attr is None:
            raise ValueError("Both edge_attr and edge_attr_reasoning are None")

        # Helper: move tensor to device of projection module (if provided), else keep as-is
        def _maybe_move(tensor: torch.Tensor, module: nn.Module):
            if tensor is None:
                return None
            if module is not None:
                dev = next(module.parameters()).device
                return tensor.to(dev)
            return tensor

        # Project if projection modules are given
        if proj_layer_stat is not None and edge_attr is not None:
            edge_attr = _maybe_move(edge_attr, proj_layer_stat)
            edge_attr = proj_layer_stat(edge_attr)

        # Methods
        method = method.lower()
        if method == 'stat':
            if edge_attr is None:
                raise ValueError("edge_attr is None but method='stat'")
            return edge_attr



    def encode_texts_with_prompt(self, texts: list, max_length: int = 256) -> torch.Tensor:
        """
        Encode a list of texts with the HF causal LM and return pooled vectors [B, H].
         It runs the model (forward with input_ids) and mean-pools
        last_hidden_state using attention_mask.
        """
        if len(texts) == 0:
            return torch.zeros((0, self.llm_hidden_size), device=self.device)

        toks = self.llm_tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        device = self.llm_model.device  # or self.device if you have it
        toks = {k: v.to(device) for k, v in toks.items()}  # move everything to GPU
        outputs = self.llm_model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"], output_hidden_states=True, return_dict=True)

        last_hidden = outputs.hidden_states[-1]  # [B, L, H] (may be float16)

        attention_mask = toks["attention_mask"].unsqueeze(-1).to(last_hidden.dtype).to(self.device)
        pooled = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1).clamp(min=1.0)
        return pooled  # [B, H]


    def _get_node_embedding_projections(self, codes_to_compute: dict = None, max_updates_per_feat: int = 20, epoch=None):
        proj = self.lamma_node_proj
        proj_device = next(proj.parameters()).device
        proj_dtype = next(proj.parameters()).dtype

        projected_parts = {}
        for feature_key in ['conditions', 'drugs', 'procedures']:
            # load cache & mask (buffers)
            cache_name = f"llm_proj_cache_{feature_key}"
            cache = getattr(self, cache_name)
            if cache is None:
                raise RuntimeError(f"{cache_name} not found")

            if (epoch==None) or (self.use_freezed_llm_for_node) or (codes_to_compute is None) or (len(codes_to_compute[feature_key]) == 0) or (epoch>self.max_epoch_to_train_node_llm) or (epoch<self.min_epoch_to_train_node_llm):
                cache_for_proj = cache.to(proj_device).to(proj_dtype)
                projected_full = proj(cache_for_proj)  # -> [N, D] (on proj_device / proj_dtype)
                emb_device = self.embeddings3[feature_key].weight.device
                emb_dtype = self.embeddings3[feature_key].weight.dtype
                projected_parts[feature_key] = projected_full.to(emb_device).to(emb_dtype)
                continue

            # get raw buffer (python list) aligned with tokenizer order
            texts_for_tokens = getattr(self, f"prompt_text_{feature_key}", None)
            if texts_for_tokens is None:
                raise RuntimeError(
                    f"prompt_text_{feature_key} not prepared; call _prapre_node_text_prompts_for_llm() in __init__")

            # compute only requested token indices and reuse cache for others
            toks = codes_to_compute[feature_key]
            compute_list = sorted(set(toks))
            k = min(max_updates_per_feat, len(compute_list))
            # here we choose random.sample but guard size
            if k == len(compute_list):
                compute_list = compute_list
            else:
                compute_list = random.sample(compute_list, k)

            # Compute embeddings for tokens in compute_list
            texts = [texts_for_tokens[t] for t in compute_list]
            raw_embs = self.encode_texts_with_prompt(texts)  # [B, H]
            raw_embs = raw_embs.to(proj_device).to(proj_dtype)
            assert raw_embs.shape[0] == len(compute_list), "encoded count mismatch"
            projected_on_cache = cache.clone()  # same device/dtype as cache
            projected_on_cache[compute_list] = raw_embs.to(projected_on_cache.device)
            getattr(self, cache_name).data.copy_(projected_on_cache)

            projected_parts[feature_key] = proj(getattr(self, cache_name).to(self.embeddings3[feature_key].weight.device).to(self.embeddings3[feature_key].weight.dtype))

        return projected_parts

    def LLM_induced_GNN(self, epoch, codes_to_compute: dict = None, max_updates_per_feat: int = 20):
        if (self.use_llm_for_node) and (epoch == self.max_epoch_to_train_node_llm) and (not self.use_freezed_llm_for_node):
            print('********* stop finetuning LLM for node: will use the last updated cache embeddings *********')
        if (self.use_llm_for_node) and (epoch == self.min_epoch_to_train_node_llm+1) and (not self.use_freezed_llm_for_node):
            print('********* start finetuning LLM for node *********')
        #random intialized mebeddings
        dx_res = self.embeddings3['conditions'].weight
        rx_res = self.embeddings3['drugs'].weight
        px_res = self.embeddings3['procedures'].weight

        if self.use_llm_for_node and (not self.init_w_freeze_llm):
            llm_projected_embeddings_for_nodes = self._get_node_embedding_projections(codes_to_compute,
                                                                                      max_updates_per_feat=max_updates_per_feat,
                                                                                      epoch=epoch)
            if not self.no_random_emb:
                # final code vector = projected_llm + residual
                dx_llm_alpha = torch.clamp(self.dx_llm_alpha, 0.0, 1.0)
                rx_llm_alpha = torch.clamp(self.rx_llm_alpha, 0.0, 1.0)
                px_llm_alpha = torch.clamp(self.px_llm_alpha, 0.0, 1.0)

                dx_emb = dx_llm_alpha * llm_projected_embeddings_for_nodes['conditions'].to(dx_res.device) + (1 - dx_llm_alpha) * dx_res
                rx_emb = rx_llm_alpha * llm_projected_embeddings_for_nodes['drugs'].to(rx_res.device) + (1 - rx_llm_alpha) * rx_res
                px_emb = px_llm_alpha * llm_projected_embeddings_for_nodes['procedures'].to(px_res.device) + (1 - px_llm_alpha) * px_res
            else:
                dx_emb = llm_projected_embeddings_for_nodes['conditions'].to(dx_res.device)
                rx_emb = llm_projected_embeddings_for_nodes['drugs'].to(rx_res.device)
                px_emb = llm_projected_embeddings_for_nodes['procedures'].to(px_res.device)

        elif self.use_llm_for_node and self.init_w_freeze_llm:
            dx_emb = self.lamma_node_proj(dx_res)
            rx_emb = self.lamma_node_proj(rx_res)
            px_emb = self.lamma_node_proj(px_res)
        else:
            dx_emb = dx_res
            rx_emb = rx_res
            px_emb = px_res

        emb_input = torch.cat([dx_emb, rx_emb, px_emb], dim=0)
        device = emb_input.device

        # --- GNN forward unchanged ---
        edge_index = self.edge_index.to(device)
        if self.use_edge_attr:
            edge_attr = self.combine_edge_features(edge_attr=self.edge_attr_stat, method=self.edge_attr_method,
                                                  proj_layer_stat=self.edge_feat_proj).to(device)
        else:
            edge_attr=None
        edge_rel_type = self.edge_rel_type.to(device)

        edge_index, edge_attr, edge_rel_type = self.make_undirected_with_metadata(edge_index, edge_attr, edge_rel_type)
        #data = Data(x=emb_input, edge_index=edge_index, edge_attr=edge_attr)

        x = emb_input
        for layer in self.gnn_layers:
            x_new = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            x = x + F.relu(x_new)
            x = self.gnn_layernorm(x)
            x = self.gnn_dropout(x)

        gnn_emb = x

        alpha = torch.clamp(self.gnn_alpha, 0.0, 1.0)
        fused = (1.0 - alpha) * emb_input + alpha * gnn_emb

        n_dx = dx_emb.shape[0]
        n_rx = rx_emb.shape[0]
        dx_new = fused[:n_dx]
        rx_new = fused[n_dx: n_dx + n_rx]
        px_new = fused[n_dx + n_rx:]

        return {'conditions': dx_new, 'drugs': rx_new, 'procedures': px_new}


    def CustomEmbeddingLookup(self, embedding_matrix, input_indices):
        batch_size, num_visits, num_codes = input_indices.shape
        input_indices_flat = input_indices.view(-1, num_codes)
        embedded = embedding_matrix[input_indices_flat]  # gather using tensor indexing
        embedded = embedded.view(batch_size, num_visits, num_codes, -1)
        return embedded

    def forward(self, kwargs, epoch=None):


        if (self.use_llm_for_node) and (self.save_last_lora) and (epoch == 150):
            self.save_lora_and_caches()

        codes_to_compute = {}
        if self.use_llm_for_node:
            # Build per-feature unique token index lists from the incoming batch
            for feature_key in self.feature_keys:
                # encode incoming batch into token indices (same as later)
                input_idxs = self.feat_tokenizers3[feature_key].batch_encode_3d(kwargs[feature_key])
                input_idxs = np.array(input_idxs, dtype=np.int64)  # shape [B, V, C]
                # flatten and unique
                flat = input_idxs.reshape(-1)
                # exclude padding and ukn idx (and optionally unk)
                flat = flat[(flat != 0)]
                valid = flat[(flat != 1)]
                if valid.size == 0:
                    codes_to_compute[feature_key] = []
                else:
                    unique_tokens = np.unique(valid).tolist()
                    codes_to_compute[feature_key] = unique_tokens

        new_embeddings = self.LLM_induced_GNN(epoch=epoch, codes_to_compute=codes_to_compute, max_updates_per_feat=self.max_updates_per_feat)

        patient_emb = []

        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            assert input_info["dim"] == 3 and input_info["type"] == str

            input_idxs = self.feat_tokenizers3[feature_key].batch_encode_3d(kwargs[feature_key])
            input_idxs = torch.tensor(input_idxs, dtype=torch.long, device=self.device)

            x = self.CustomEmbeddingLookup(new_embeddings[feature_key], input_idxs)
            x = torch.sum(x, dim=2)
            pad_idx = self.feat_tokenizers3[feature_key].get_padding_index() #self.feat_tokenizers3[feature_key].vocabulary("<pad>")
            mask = torch.any(input_idxs != pad_idx, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        y_true = self.prepare_labels(kwargs[self.label_key], self.get_label_tokenizer())
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true}
