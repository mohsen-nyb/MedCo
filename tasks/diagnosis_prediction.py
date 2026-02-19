from pyhealth.medcode import InnerMap
from pyhealth.medcode import CrossMap
import numpy as np
import pandas as pd
import random

def sequential_diagnosis_prediction_mimic3(patient, dx_converter,px_converter):

    '''Sequential diagnosis prediction aims at predicting the diagnosis set of the next visit given the diagnosis,
     procedure, and drug information from the past visits.'''

    samples = []
    ccs_samples = []

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = []

    sequential_conditions_ccs = []
    sequential_procedures_ccs = []

    sequential_visit_index=[]

    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        next_visit = patient[idx + 1]

        # step 1: diganosis as label
        diagnosis_label = next_visit.get_code_list(table="DIAGNOSES_ICD")
        if '71970' in diagnosis_label:
            diagnosis_label.remove('71970')
            diagnosis_label.append('7197')
        diagnosis_label_ccs = list(set(dx_converter(diagnosis_label)))

        # step 2: define features (current visit)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        if '71970' in conditions:
            conditions.remove('71970')
            conditions.append('7197')
        conditions_ccs = list(set(dx_converter(conditions)))

        procedures = visit.get_code_list(table="PROCEDURES_ICD")


        if '3608' in procedures:
            procedures.remove('3608')
        if '3618' in procedures:
            procedures.remove('3618')
        procedures_ccs = list(set(px_converter(procedures)))


        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        sequential_conditions_ccs.append(conditions_ccs)
        sequential_procedures_ccs.append(procedures_ccs)

        sequential_conditions.append(conditions)
        sequential_procedures.append(procedures)
        sequential_drugs.append(drugs)

        sequential_visit_index.append([idx])

        if (len(diagnosis_label) == 0) or (len(diagnosis_label_ccs) == 0): continue

        # step 4: assemble the samples
        samples.append(
            {
                "visit_id": next_visit.visit_id,
                "visit_index_list": sequential_visit_index.copy(),
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "drugs": sequential_drugs.copy(),
                "label": diagnosis_label,
            }
        )

        ccs_samples.append(
            {
                "visit_id": next_visit.visit_id,
                "visit_index_list": sequential_visit_index.copy(),
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions_ccs.copy(),
                "procedures": sequential_procedures_ccs.copy(),
                "drugs": sequential_drugs.copy(),
                "label": diagnosis_label_ccs,
            }
        )
    return samples, ccs_samples

import numpy as np

def _clean_codes(codes):
    """Remove None/NaN and cast everything to a normalized string."""
    out = []
    if codes is None:
        return out
    for c in codes:
        # drop None / NaN
        if c is None:
            continue
        if isinstance(c, float) and np.isnan(c):
            continue

        # normalize to string
        if isinstance(c, float):
            # 7197.0 -> "7197"
            if c.is_integer():
                c = str(int(c))
            else:
                c = str(c)
        else:
            c = str(c)

        c = c.strip()
        if c == "" or c.lower() == "nan":
            continue

        # optional: remove dots if your vocab/mappers expect dotless codes
        # (matches your '71970' -> '7197' style)
        #c = c.replace(".", "")

        out.append(c)
    return out
def sequential_diagnosis_prediction_mimic4(patient, dx_converter,px_converter,
                                           dx_mapper, px_mapper, icd9proc, icd9cm):
    ''' Sequential diagnosis prediction aims at predicting the diagnosis set of the next visit given the diagnosis,
     procedure, and drug information from the past visits. '''


    samples = []
    ccs_samples = []

    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = []

    sequential_conditions_ccs = []
    sequential_procedures_ccs = []

    sequential_visit_index=[]

    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        next_visit = patient[idx + 1]

        # step 1: diganosis as label
        diagnosis_label = next_visit.get_code_list(table="diagnoses_icd")
        if '71970' in diagnosis_label:
            diagnosis_label.remove('71970')
            diagnosis_label.append('7197')

        diagnosis_label_9=[]
        diagnosis_label_10=[]
        for code in diagnosis_label:
            if code in icd9cm:
                diagnosis_label_9.append(code)
            else:
                diagnosis_label_10.append(code)
        if len(diagnosis_label_10)!=0:
            mapped_dx_9 = np.array(dx_mapper.map(diagnosis_label_10))
            new_icd9 = mapped_dx_9[(mapped_dx_9 != None) & (mapped_dx_9 != 'NoDx')]
            new_icd9 = [code for code in new_icd9.tolist() if code in icd9cm]
            new_diagnosis_label = list(set(diagnosis_label_9 + new_icd9))
        else:
            new_diagnosis_label = diagnosis_label_9

        diagnosis_label_ccs = list(set(dx_converter(new_diagnosis_label)))
        diagnosis_label_ccs = _clean_codes(diagnosis_label_ccs)

        # step 2: define features (current visit)
        conditions = visit.get_code_list(table="diagnoses_icd")
        if '71970' in conditions:
            conditions.remove('71970')
            conditions.append('7197')

        conditions_9 = []
        conditions_10 = []
        for code in conditions:
            if code in icd9cm:
                conditions_9.append(code)
            else:
                conditions_10.append(code)
        if len(conditions_10) != 0:
            mapped_conditions_9 = np.array(dx_mapper.map(conditions_10))
            new_conditions_icd9 = mapped_conditions_9[(mapped_conditions_9 != None) & (mapped_conditions_9 != 'NoDx')]
            new_conditions_icd9 = [code for code in new_conditions_icd9.tolist() if code in icd9cm]
            new_conditions = list(set(conditions_9 + new_conditions_icd9))
        else:
            new_conditions = conditions_9

        conditions_ccs = list(set(dx_converter(new_conditions)))
        conditions_ccs = _clean_codes(conditions_ccs)

        procedures = visit.get_code_list(table="procedures_icd")
        if '3608' in procedures:
            procedures.remove('3608')
        if '3618' in procedures:
            procedures.remove('3618')

        procedures_9=[]
        procedures_10=[]
        for code in procedures:
            if code in icd9proc:
                procedures_9.append(code)
            else:
                procedures_10.append(code)
        if len(procedures_10) !=0:
            mapped_procedures_9 = np.array(px_mapper.map(procedures_10))
            new_procedures_icd9 = mapped_procedures_9[(mapped_procedures_9 != None) & (mapped_procedures_9 != 'NoI9')]
            new_procedures_icd9 = [code for code in new_procedures_icd9.tolist() if code in icd9proc]
            new_procedures = list(set(procedures_9 + new_procedures_icd9))
        else:
            new_procedures = procedures_9
        procedures_ccs = list(set(px_converter(new_procedures)))
        procedures_ccs = _clean_codes(procedures_ccs)

        drugs = visit.get_code_list(table="prescriptions")

        sequential_conditions_ccs.append(conditions_ccs)
        sequential_procedures_ccs.append(procedures_ccs)

        sequential_conditions.append(new_conditions)
        sequential_procedures.append(new_procedures)
        sequential_drugs.append(drugs)

        sequential_visit_index.append([idx])

        if (len(new_diagnosis_label) == 0) or (len(diagnosis_label_ccs) == 0): continue

        # step 4: assemble the samples
        samples.append(
            {
                "visit_id": next_visit.visit_id,
                "visit_index_list": sequential_visit_index.copy(),
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "drugs": sequential_drugs.copy(),
                "label": new_diagnosis_label,
            }
        )

        ccs_samples.append(
            {
                "visit_id": next_visit.visit_id,
                "visit_index_list": sequential_visit_index.copy(),
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions_ccs.copy(),
                "procedures": sequential_procedures_ccs.copy(),
                "drugs": sequential_drugs.copy(),
                "label": diagnosis_label_ccs,
            }
        )
    return samples, ccs_samples



def diagnosis_prediction(patient):
    """Diagnosis Prediction aims at predicting the diagnosis set of the next visit given the diagnosis,
     procedure, and drug information from the current visit."""
    samples = []

    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        next_visit = patient[idx + 1]

        # step 1: diganosis as label
        diagnosis_label = next_visit.get_code_list(table="DIAGNOSES_ICD")

        # step 2: define features
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # step 3: exclusion criteria: visits without drug
        if len(diagnosis_label) == 0: continue

        # step 4: assemble the samples
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": diagnosis_label,
            }
        )
    return samples




def sequential_diagnosis_prediction_levels(patient, icd9cm, icd9cmproc, atc, CCS_mapping, ccs_label=False):

    '''Sequential diagnosis prediction aims at predicting the diagnosis set of the next visit given the diagnosis,
     procedure, and drug information from the past visits.'''


    samples1 = []
    samples2 = []
    samples3 = []

    sequential_conditions1 = []
    sequential_conditions2 = []
    sequential_conditions3 = []

    sequential_procedures1 = []
    sequential_procedures2 = []
    sequential_procedures3 = []


    sequential_drugs1 = []
    sequential_drugs2 = []
    sequential_drugs3 = []



    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        next_visit = patient[idx + 1]

        # step 1: diganosis as label
        diagnosis_label = next_visit.get_code_list(table="DIAGNOSES_ICD")
        if '71970' in diagnosis_label:
            diagnosis_label.remove('71970')
            diagnosis_label.append('7197')

        #diagnosis_label3=[]
        diagnosis_label2=[]
        diagnosis_label1=[]

        for code in diagnosis_label:
            parents = icd9cm.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()

            diagnosis_label1.append(parents_child[0])
            diagnosis_label2.append(parents_child[1])
            #diagnosis_label3.append(parents_child[2])



        # step 2: define features (current visit)
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        if '71970' in conditions:
            conditions.remove('71970')
            conditions.append('7197')
        #conditions3 = []
        conditions2 = []
        conditions1 = []
        for code in conditions:
            parents = icd9cm.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()

            conditions1.append(parents_child[0])
            conditions2.append(parents_child[1])
            #conditions3.append(parents_child[2])

        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        if '3601' in procedures:
            procedures.remove('3601')
            procedures.append('3603')
            procedures = list(set(procedures))

        if '3602' in procedures:
            procedures.remove('3602')
            procedures.append('3603')
            procedures = list(set(procedures))

        if '3605' in procedures:
            procedures.remove('3605')
            procedures.append('3603')
            procedures = list(set(procedures))

        if '3608' in procedures:
            procedures.remove('3608')
        if '3618' in procedures:
            procedures.remove('3618')

        #procedures3 = []
        procedures2 = []
        procedures1 = []
        for code in procedures:
            parents = icd9cmproc.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()
            procedures1.append(parents_child[0])
            procedures2.append(parents_child[1])
            #procedures3.append(parents_child[2])


        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        #drugs3 = []
        drugs2 = []
        drugs1 = []
        for code in drugs:
            parents = atc.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()

            drugs1.append(parents_child[0])
            drugs2.append(parents_child[1])
            #drugs3.append(parents_child[2])



        sequential_conditions3.append(conditions)
        sequential_conditions2.append(list(set(conditions2)))
        sequential_conditions1.append(list(set(conditions1)))


        sequential_procedures3.append(procedures)
        sequential_procedures2.append(list(set(procedures2)))
        sequential_procedures1.append(list(set(procedures1)))


        sequential_drugs3.append(drugs)
        sequential_drugs2.append(list(set(drugs2)))
        sequential_drugs1.append(list(set(drugs1)))



        # step 3: exclusion criteria: visits without diagnosis codes
        if len(diagnosis_label) == 0: continue


        if ccs_label==True:
            # diagnosis_label_maped1 = []
            # diagnosis_label_maped2 = []
            diagnosis_label_maped3 = []

            # for code in diagnosis_label1:
            #     diagnosis_label_maped1.extend(CCS_mapping.map(code))
            # for code in diagnosis_label2:
            #     diagnosis_label_maped2.extend(CCS_mapping.map(code))
            for code in diagnosis_label:
                diagnosis_label_maped3.extend(CCS_mapping.map(code))

            diagnosis_label1 = diagnosis_label_maped3
            diagnosis_label2 = diagnosis_label_maped3
            diagnosis_label = diagnosis_label_maped3

            if len(diagnosis_label) == 0: continue


        samples3.append(
            {
                "visit_id": next_visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions3.copy(),
                "procedures": sequential_procedures3.copy(),
                "drugs": sequential_drugs3.copy(),
                "label": diagnosis_label,
            }
        )
        samples2.append(
            {
                "visit_id": next_visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions2.copy(),
                "procedures": sequential_procedures2.copy(),
                "drugs": sequential_drugs2.copy(),
                "label": diagnosis_label2,
            }
        )

        samples1.append(
            {
                "visit_id": next_visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions1.copy(),
                "procedures": sequential_procedures1.copy(),
                "drugs": sequential_drugs1.copy(),
                "label": diagnosis_label1,
            }
        )

    return samples1, samples2, samples3



def sequential_diagnosis_prediction_levels_mimic4(patient, dx_mapper, px_mapper, icd9cm, icd9cmproc, atc, CCS_mapping,
                                                  ccs_label=False):
    '''Sequential diagnosis prediction aims at predicting the diagnosis set of the next visit given the diagnosis,
     procedure, and drug information from the past visits.'''

    samples1 = []
    samples2 = []
    samples3 = []

    sequential_conditions1 = []
    sequential_conditions2 = []
    sequential_conditions3 = []

    sequential_procedures1 = []
    sequential_procedures2 = []
    sequential_procedures3 = []

    sequential_drugs1 = []
    sequential_drugs2 = []
    sequential_drugs3 = []

    for idx, visit in enumerate(patient):
        if idx == len(patient) - 1: break
        next_visit = patient[idx + 1]

        # step 1: diganosis as label
        diagnosis_label = next_visit.get_code_list(table="diagnoses_icd")
        if '71970' in diagnosis_label:
            diagnosis_label.remove('71970')
            diagnosis_label.append('7197')

        # step 2: define features (current visit)
        conditions = visit.get_code_list(table="diagnoses_icd")
        if '71970' in conditions:
            conditions.remove('71970')
            conditions.append('7197')

        conditions_9 = []
        conditions_10 = []
        for code in conditions:
            if code in icd9cm:
                conditions_9.append(code)
            else:
                conditions_10.append(code)
        if len(conditions_10) != 0:
            mapped_conditions_9 = np.array(dx_mapper.map(conditions_10))
            new_conditions_icd9 = mapped_conditions_9[(mapped_conditions_9 != None) & (mapped_conditions_9 != 'NoDx')]
            new_conditions_icd9 = [code for code in new_conditions_icd9.tolist() if code in icd9cm]
            new_conditions = list(set(conditions_9 + new_conditions_icd9))
        else:
            new_conditions = conditions_9

        new_conditions2 = []
        new_conditions1 = []
        for code in new_conditions:
            parents = icd9cm.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()
            new_conditions1.append(parents_child[0])
            new_conditions2.append(parents_child[1])

        procedures = visit.get_code_list(table="procedures_icd")
        if '3601' in procedures:
            procedures.remove('3601')
            procedures.append('3603')
            procedures = list(set(procedures))

        if '3602' in procedures:
            procedures.remove('3602')
            procedures.append('3603')
            procedures = list(set(procedures))

        if '3605' in procedures:
            procedures.remove('3605')
            procedures.append('3603')
            procedures = list(set(procedures))

        if '3608' in procedures:
            procedures.remove('3608')
        if '3618' in procedures:
            procedures.remove('3618')

        procedures_9 = []
        procedures_10 = []
        for code in procedures:
            if code in icd9cmproc:
                procedures_9.append(code)
            else:
                procedures_10.append(code)
        if len(procedures_10) != 0:
            mapped_procedures_9 = np.array(px_mapper.map(procedures_10))
            new_procedures_icd9 = mapped_procedures_9[(mapped_procedures_9 != None) & (mapped_procedures_9 != 'NoI9')]
            new_procedures_icd9 = [code for code in new_procedures_icd9.tolist() if code in icd9cmproc]
            new_procedures = list(set(procedures_9 + new_procedures_icd9))
        else:
            new_procedures = procedures_9

        new_procedures2 = []
        new_procedures1 = []
        for code in new_procedures:
            parents = icd9cmproc.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()
            new_procedures1.append(parents_child[0])
            new_procedures2.append(parents_child[1])

        drugs = visit.get_code_list(table="prescriptions")
        drugs2 = []
        drugs1 = []
        for code in drugs:
            parents = atc.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()
            drugs1.append(parents_child[0])
            drugs2.append(parents_child[1])

        sequential_conditions3.append(new_conditions)
        sequential_conditions2.append(list(set(new_conditions2)))
        sequential_conditions1.append(list(set(new_conditions1)))

        sequential_procedures3.append(new_procedures)
        sequential_procedures2.append(list(set(new_procedures2)))
        sequential_procedures1.append(list(set(new_procedures1)))

        sequential_drugs3.append(drugs)
        sequential_drugs2.append(list(set(drugs2)))
        sequential_drugs1.append(list(set(drugs1)))

        # step 3: exclusion criteria: visits without diagnosis codes
        if len(diagnosis_label) == 0: continue
        diagnosis_label_9 = []
        diagnosis_label_10 = []
        for code in diagnosis_label:
            if code in icd9cm:
                diagnosis_label_9.append(code)
            else:
                diagnosis_label_10.append(code)
        if len(diagnosis_label_10) != 0:
            mapped_dx_9 = np.array(dx_mapper.map(diagnosis_label_10))
            new_icd9 = mapped_dx_9[(mapped_dx_9 != None) & (mapped_dx_9 != 'NoDx')]
            new_icd9 = [code for code in new_icd9.tolist() if code in icd9cm]
            new_diagnosis_label = list(set(diagnosis_label_9 + new_icd9))
        else:
            new_diagnosis_label = diagnosis_label_9

        new_diagnosis_label2 = []
        new_diagnosis_label1 = []
        for code in new_diagnosis_label:
            parents = icd9cm.get_ancestors(code)
            parents_child = [code] + parents
            parents_child.reverse()
            new_diagnosis_label1.append(parents_child[0])
            new_diagnosis_label2.append(parents_child[1])

        if ccs_label == True:
            # diagnosis_label_maped1 = []
            # diagnosis_label_maped2 = []
            diagnosis_label_maped3 = []

            # for code in diagnosis_label1:
            #     diagnosis_label_maped1.extend(CCS_mapping.map(code))
            # for code in diagnosis_label2:
            #     diagnosis_label_maped2.extend(CCS_mapping.map(code))
            for code in new_diagnosis_label:
                diagnosis_label_maped3.extend(CCS_mapping.map(code))

            new_diagnosis_label1 = diagnosis_label_maped3
            new_diagnosis_label2 = diagnosis_label_maped3
            new_diagnosis_label = diagnosis_label_maped3


        if len(new_diagnosis_label) == 0: continue


        samples3.append(
            {
                "visit_id": next_visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions3.copy(),
                "procedures": sequential_procedures3.copy(),
                "drugs": sequential_drugs3.copy(),
                "label": new_diagnosis_label,
            }
        )
        samples2.append(
            {
                "visit_id": next_visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions2.copy(),
                "procedures": sequential_procedures2.copy(),
                "drugs": sequential_drugs2.copy(),
                "label": new_diagnosis_label,
            }
        )

        samples1.append(
            {
                "visit_id": next_visit.visit_id,
                "patient_id": patient.patient_id,
                # the following keys can be the "feature_keys" or "label_key" for initializing downstream ML model
                "conditions": sequential_conditions1.copy(),
                "procedures": sequential_procedures1.copy(),
                "drugs": sequential_drugs1.copy(),
                "label": new_diagnosis_label,
            }
        )

    return samples1, samples2, samples3




