"""下载失败的 benchmark 数据集。"""
from datasets import load_dataset, concatenate_datasets, DatasetDict
import os

save_dir = 'data/benchmark'

# piqa
print('=== piqa ===')
try:
    ds = load_dataset('ybisk/piqa', trust_remote_code=True)
    for s in ds:
        print(f'  {s}: {len(ds[s])}')
    ds.save_to_disk(os.path.join(save_dir, 'piqa'))
    print('  Saved')
except Exception as e:
    print(f'  FAILED: {e}')

# siqa
print('=== siqa ===')
try:
    ds = load_dataset('allenai/social_i_qa', trust_remote_code=True)
    for s in ds:
        print(f'  {s}: {len(ds[s])}')
    ds.save_to_disk(os.path.join(save_dir, 'siqa'))
    print('  Saved')
except Exception as e:
    print(f'  FAILED: {e}')

# cmmlu - 逐科目下载合并
print('=== cmmlu ===')
try:
    subjects = [
        'agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy',
        'business_ethics', 'chinese_civil_service_exam', 'chinese_driving_rule',
        'chinese_food_culture', 'chinese_foreign_policy', 'chinese_history',
        'chinese_literature', 'chinese_teacher_qualification', 'clinical_knowledge',
        'college_actuarial_science', 'college_education',
        'college_engineering_hydrology', 'college_law', 'college_mathematics',
        'college_medicine_and_public_health', 'computer_science',
        'conceptual_physics', 'construction_project_management', 'economics',
        'education', 'electrical_engineering', 'elementary_chinese',
        'elementary_commonsense', 'elementary_information_and_technology',
        'elementary_mathematics', 'ethnology', 'food_science', 'genetics',
        'global_facts', 'high_school_biology', 'high_school_chemistry',
        'high_school_geography', 'high_school_mathematics', 'high_school_physics',
        'high_school_politics', 'human_sexuality', 'international_law',
        'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical',
        'machine_learning', 'management', 'marketing', 'marxist_theory',
        'modern_chinese', 'nutrition', 'philosophy', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_study', 'sociology', 'sports_science',
        'traditional_chinese_medicine', 'virology', 'world_history',
        'world_religions',
    ]
    all_test, all_dev = [], []
    for subj in subjects:
        try:
            d = load_dataset('haonan-li/cmmlu', subj, trust_remote_code=True)
            if 'test' in d:
                all_test.append(d['test'])
            if 'dev' in d:
                all_dev.append(d['dev'])
        except Exception:
            pass
    print(f'  Loaded {len(all_test)} subjects')
    merged = DatasetDict()
    if all_test:
        merged['test'] = concatenate_datasets(all_test)
    if all_dev:
        merged['dev'] = concatenate_datasets(all_dev)
    for s in merged:
        print(f'  {s}: {len(merged[s])}')
    merged.save_to_disk(os.path.join(save_dir, 'cmmlu'))
    print('  Saved')
except Exception as e:
    print(f'  FAILED: {e}')

# ceval - 逐科目下载合并
print('=== ceval ===')
try:
    ceval_subjects = [
        'accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine',
        'business_administration', 'chinese_language_and_literature',
        'civil_servant', 'clinical_medicine', 'college_chemistry',
        'college_economics', 'college_physics', 'college_programming',
        'computer_architecture', 'computer_network', 'discrete_mathematics',
        'education_science', 'electrical_engineer',
        'environmental_impact_assessment_engineer', 'fire_engineer',
        'high_school_biology', 'high_school_chemistry', 'high_school_chinese',
        'high_school_geography', 'high_school_history',
        'high_school_mathematics', 'high_school_physics',
        'high_school_politics', 'ideological_and_moral_cultivation', 'law',
        'legal_professional', 'logic', 'mao_zedong_thought', 'marxism',
        'metrology_engineer', 'middle_school_biology',
        'middle_school_chemistry', 'middle_school_geography',
        'middle_school_history', 'middle_school_mathematics',
        'middle_school_physics', 'middle_school_politics',
        'modern_chinese_history', 'operating_system', 'physician',
        'plant_protection', 'probability_and_statistics',
        'professional_tour_guide', 'sports_science', 'tax_accountant',
        'teacher_qualification', 'urban_and_rural_planner',
        'veterinary_medicine',
    ]
    all_val, all_dev = [], []
    for subj in ceval_subjects:
        try:
            d = load_dataset('ceval/ceval-exam', subj, trust_remote_code=True)
            if 'val' in d:
                all_val.append(d['val'])
            if 'dev' in d:
                all_dev.append(d['dev'])
        except Exception:
            pass
    print(f'  Loaded {len(all_val)} subjects')
    merged = DatasetDict()
    if all_val:
        merged['val'] = concatenate_datasets(all_val)
    if all_dev:
        merged['dev'] = concatenate_datasets(all_dev)
    for s in merged:
        print(f'  {s}: {len(merged[s])}')
    merged.save_to_disk(os.path.join(save_dir, 'ceval'))
    print('  Saved')
except Exception as e:
    print(f'  FAILED: {e}')

# c3
print('=== c3 ===')
try:
    ds = load_dataset('clue', 'c3', trust_remote_code=True)
    for s in ds:
        print(f'  {s}: {len(ds[s])}')
    ds.save_to_disk(os.path.join(save_dir, 'c3'))
    print('  Saved')
except Exception as e:
    print(f'  FAILED: {e}')

print('\nAll done.')
