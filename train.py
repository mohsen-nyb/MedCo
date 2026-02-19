from utils.data import customized_set_task_mimic4
from tasks.diagnosis_prediction import sequential_diagnosis_prediction_mimic4
from pyhealth.datasets import split_by_patient, get_dataloader
from model import CoMed
#from pyhealth.trainer import Trainer
from trainer import Trainer
import torch
import numpy as np
from pyhealth.datasets import MIMIC4Dataset
import random
from utils.eval_test import evaluate, get_group_labels1, calculate_confidence_interval
import os
import shutil
import logging
from datetime import datetime


def cleanup_trainer_run(trainer):
    # Close file handlers to avoid Windows file-lock issues
    root_logger = logging.getLogger()  # global root
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            root_logger.removeHandler(h)

    # Also try the trainer module logger name if present
    tlogger = logging.getLogger("trainer")
    for h in list(tlogger.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            tlogger.removeHandler(h)

    # Remove the experiment directory
    if getattr(trainer, "exp_path", None):
        shutil.rmtree(trainer.exp_path, ignore_errors=True)


def nfold_experiment(mimic3sample, mimic3sample_whole, epochs, print_results=True, record_results=True):

    data = mimic3sample.samples
    co_occurrence_counts, groups1 = get_group_labels1(data)

    seeds = [42, 123, 51, 32, 12]

    list_top_k = [3,5,7,10,15,20,30]
    metrics_dict = {'roc_auc_samples': [], 'pr_auc_samples': [], 'f1_samples': []}

    for group_name in groups1.keys():
        metrics_dict[f'roc_auc_samples_{group_name}'] = []
        metrics_dict[f'pr_auc_samples_{group_name}'] = []


    for k in list_top_k:
        metrics_dict[f'acc_at_k={k}'] = []
        metrics_dict[f'hit_at_k={k}'] = []
        for group_name in groups1.keys():
            metrics_dict[f'Group_acc_at_k={k}@' + group_name] = []
            metrics_dict[f'Group_hit_at_k={k}@' + group_name] = []


    for seed in seeds:
        print(f'----------------------seed:{seed}-----------------------')

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Set seed for CUDA operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [train_ratio, val_ratio, test_ratio], seed=seed)
        # create dataloaders (torch.data.DataLoader)
        train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)
        print('--data loaders created.')


        print('preprocessing done!')

        # Stage 3: define model
        device = "cuda:0"

        model = CoMed(
            dataset=mimic3sample_whole,
            feature_keys=["conditions", "drugs", "procedures"],
            label_key="label",
            mode="multilabel",
            embedding_dim=256,
            dropout = 0.5,
            nheads=1,
            nlayers=1,
            # globe GNN hyperparams
            nlayers_gnn= 2,
            n_gat_heads= 1,
            gnn_dropout= 0.4,
            use_llm_for_node = True,
            use_freezed_llm_for_node=True,
            no_random_emb=False,
            init_w_freeze_llm=False,
            use_edge_attr = True,
            llm_node_path = "../saved_files/mimic4/KG_openai/nodes/node_text_embeddings_input_final.parquet",
            llm_residual_scale = 0.01,
            edge_attr_method = 'cat',
            # llm params
            llm_name="meta-llama/Llama-3.2-1B",
            lora_r=8,
            lora_alpha=32,
            max_updates_per_feat=10,
            max_epoch_to_train_node_llm=150,
            min_epoch_to_train_node_llm=-1,
            device = device,
            seed = seed,
        )

        model.to(device)

        # Stage 4: model training
        # ---- Unique exp_name per seed/run ----
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = (
            f"KGLLM_freeze"
            f"_seed{seed}_{run_tag}"
        )
        # ---- Trainer with easy fix: enable_logging=True ----
        trainer = Trainer(
            model=model,
            checkpoint_path=None,
            metrics=['roc_auc_samples', 'pr_auc_samples', 'f1_samples'],
            enable_logging=True,                      # IMPORTANT
            output_path="../tmp_runs",    # temp folder
            exp_name=exp_name,
            device=device
        )

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            optimizer_class =  torch.optim.Adam,
            optimizer_params = {"lr": 1e-3},
            weight_decay=0.0,
            monitor="pr_auc_samples",
            monitor_criterion='max',
            load_best_model_at_last=True
        )


        all_metrics = [

            "pr_auc_samples",
            "roc_auc_samples",
            "f1_samples",
        ]

        y_true, y_prob, loss = trainer.inference(test_loader)

        result = evaluate(y_true, y_prob, co_occurrence_counts, groups1, list_top_k=list_top_k, all_metrics=all_metrics)

        print ('\n', result)

        metrics_dict['pr_auc_samples'].append(result['pr_auc_samples'])
        metrics_dict['roc_auc_samples'].append(result['roc_auc_samples'])
        metrics_dict['f1_samples'].append(result['f1_samples'])

        for group_name in groups1.keys():
            metrics_dict[f'roc_auc_samples_{group_name}'].append(result[f'roc_auc_samples_{group_name}'])
            metrics_dict[f'pr_auc_samples_{group_name}'].append(result[f'pr_auc_samples_{group_name}'])


        for k in list_top_k:
            metrics_dict[f'acc_at_k={k}'].append(result[f'acc_at_k={k}'])
            metrics_dict[f'hit_at_k={k}'].append(result[f'hit_at_k={k}'])
            for group_name in groups1.keys():
                metrics_dict[f'Group_acc_at_k={k}@' + group_name].append(result[f'Group_acc_at_k={k}@' + group_name])
                metrics_dict[f'Group_hit_at_k={k}@' + group_name].append(result[f'Group_hit_at_k={k}@' + group_name])


        # ---- Cleanup run artifacts BEFORE next seed ----
        cleanup_trainer_run(trainer)
    if print_results:
        print()
        print('mean pr_auc_samples:', np.mean(metrics_dict['pr_auc_samples']))
        print('max pr_auc_samples:', np.max(metrics_dict['pr_auc_samples']))
        print('min pr_auc_samples:', np.min(metrics_dict['pr_auc_samples']))
        print('CI pr_auc_samples:', calculate_confidence_interval(metrics_dict['pr_auc_samples']))


        print()

        print('mean roc_auc_samples:', np.mean(metrics_dict['roc_auc_samples']))
        print('max roc_auc_samples:', np.max(metrics_dict['roc_auc_samples']))
        print('min roc_auc_samples:', np.min(metrics_dict['roc_auc_samples']))
        print('CI roc_auc_samples:', calculate_confidence_interval(metrics_dict['roc_auc_samples']))
        print()

        print('mean f1_samples:', np.mean(metrics_dict['f1_samples']))
        print('max f1_samples:', np.max(metrics_dict['f1_samples']))
        print('min f1_samples:', np.min(metrics_dict['f1_samples']))
        print('CI f1_samples:', calculate_confidence_interval(metrics_dict['f1_samples']))
        print()

        for group_name in groups1:
            print()
            print(f'mean pr_auc_samples_{group_name}:', np.mean(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'max pr_auc_samples_{group_name}:', np.max(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'min pr_auc_samples_{group_name}:', np.min(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'CI pr_auc_samples_{group_name}:', calculate_confidence_interval(metrics_dict[f'pr_auc_samples_{group_name}']))
            print()

            print(f'mean roc_auc_samples_{group_name}:', np.mean(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'max roc_auc_samples_{group_name}:', np.max(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'min roc_auc_samples_{group_name}:', np.min(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'CI roc_auc_samples_{group_name}:', calculate_confidence_interval(metrics_dict[f'roc_auc_samples_{group_name}']))
            print()


        for k in list_top_k:
            print('------------------------------------------')

            print(f'mean acc_at_k={k}:', np.mean(metrics_dict[f'acc_at_k={k}']))
            print(f'max acc_at_k={k}:', np.max(metrics_dict[f'acc_at_k={k}']))
            print(f'min acc_at_k={k}:', np.min(metrics_dict[f'acc_at_k={k}']))
            print(f'CI acc_at_k={k}:', calculate_confidence_interval(metrics_dict[f'acc_at_k={k}']))
            print()

            print(f'mean hit_at_k={k}:', np.mean(metrics_dict[f'hit_at_k={k}']))
            print(f'max hit_at_k={k}:', np.max(metrics_dict[f'hit_at_k={k}']))
            print(f'min hit_at_k={k}:', np.min(metrics_dict[f'hit_at_k={k}']))
            print(f'CI hit_at_k={k}:', calculate_confidence_interval(metrics_dict[f'hit_at_k={k}']))
            print()

            for group_name in groups1:

                print(f'mean Group_acc_at_k={k}@{group_name}:', np.mean(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'max Group_acc_at_k={k}@{group_name}:', np.max(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'min Group_acc_at_k={k}@{group_name}:', np.min(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'CI Group_acc_at_k={k}@{group_name}:', calculate_confidence_interval(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print()

                print(f'mean Group_hit_at_k={k}@{group_name}:', np.mean(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'max Group_hit_at_k={k}@{group_name}:', np.max(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'min Group_hit_at_k={k}@{group_name}:', np.min(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'CI Group_hit_at_k={k}@{group_name}:',calculate_confidence_interval(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print()

    if record_results:
        os.makedirs('results/', exist_ok=True)
        with open(f'results/MedC0_gat_{ds_size_ratio}.txt', 'w') as file:
            file.write('\n')
            file.write(f'mean pr_auc_samples: {np.mean(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'max pr_auc_samples: {np.max(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'min pr_auc_samples: {np.min(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'CI pr_auc_samples: {calculate_confidence_interval(metrics_dict["pr_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean roc_auc_samples: {np.mean(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'max roc_auc_samples: {np.max(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'min roc_auc_samples: {np.min(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'CI roc_auc_samples: {calculate_confidence_interval(metrics_dict["roc_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean f1_samples: {np.mean(metrics_dict["f1_samples"])}\n')
            file.write(f'max f1_samples: {np.max(metrics_dict["f1_samples"])}\n')
            file.write(f'min f1_samples: {np.min(metrics_dict["f1_samples"])}\n')
            file.write(f'CI f1_samples: {calculate_confidence_interval(metrics_dict["f1_samples"])}\n')
            file.write('\n')

            for group_name in groups1:
                file.write('\n')
                file.write(f'mean pr_auc_samples_{group_name}: {np.mean(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(f'max pr_auc_samples_{group_name}: {np.max(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(f'min pr_auc_samples_{group_name}: {np.min(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(f'CI pr_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write('\n')

                file.write(f'mean roc_auc_samples_{group_name}: {np.mean(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(f'max roc_auc_samples_{group_name}: {np.max(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(f'min roc_auc_samples_{group_name}: {np.min(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(f'CI roc_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write('\n')


            for k in list_top_k:
                file.write('------------------------------------------\n')

                file.write(f'mean acc_at_k={k}: {np.mean(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'max acc_at_k={k}: {np.max(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'min acc_at_k={k}: {np.min(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'CI acc_at_k={k}: {calculate_confidence_interval(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write('\n')

                file.write(f'mean hit_at_k={k}: {np.mean(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'max hit_at_k={k}: {np.max(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'min hit_at_k={k}: {np.min(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'CI hit_at_k={k}: {calculate_confidence_interval(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write('\n')

                for group_name in groups1:
                    file.write(
                        f'mean Group_acc_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_acc_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_acc_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_acc_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write('\n')

                    file.write('------------------------------------------\n')

                    file.write(
                        f'mean Group_hit_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_hit_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_hit_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_hit_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write('\n')

    return




train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
batch_size = 128


mimic4_ds = MIMIC4Dataset(
    root="datasets/MIMIC_IV/hosp/",
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 4}})},
)
print('--mimic-IV loaded.')

mimic3sample = customized_set_task_mimic4(dataset=mimic4_ds,
                                          task_fn=sequential_diagnosis_prediction_mimic4,
                                          ccs_label=True,
                                          seed=42)

mimic3sample_whole = customized_set_task_mimic4(dataset=mimic4_ds,
                                          task_fn=sequential_diagnosis_prediction_mimic4,
                                          ccs_label=True,
                                          seed=42)

print('--datasets created.')
print(mimic3sample.stat())


nfold_experiment(mimic3sample, mimic3sample_whole, epochs=100)



