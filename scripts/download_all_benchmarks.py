"""下载所有 benchmark 数据集，不兼容的用 HF API 直接下载 parquet/csv。"""
import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from huggingface_hub import hf_hub_download, list_repo_files

save_dir = 'data/benchmark'
os.makedirs(save_dir, exist_ok=True)


def try_load(name, repo, subset=None, **kwargs):
    """尝试多种方式下载。"""
    out = os.path.join(save_dir, name)
    if os.path.exists(out) and len(os.listdir(out)) > 1:
        print(f'  [SKIP] {name} already exists')
        return True

    # 方法1: 标准 load_dataset
    try:
        if subset:
            ds = load_dataset(repo, subset, trust_remote_code=True, **kwargs)
        else:
            ds = load_dataset(repo, trust_remote_code=True, **kwargs)
        for s in ds:
            print(f'  {s}: {len(ds[s])}')
        ds.save_to_disk(out)
        print(f'  Saved to {out}')
        return True
    except Exception as e:
        print(f'  Method1 failed: {e}')

    # 方法2: 下载 parquet 文件
    try:
        files = list_repo_files(repo, repo_type='dataset')
        parquets = [f for f in files if f.endswith('.parquet')]
        if parquets:
            os.makedirs(out, exist_ok=True)
            for pf in parquets:
                local = hf_hub_download(repo, pf, repo_type='dataset',
                                        local_dir=os.path.join(out, 'raw'))
                print(f'  Downloaded {pf}')
            # 尝试从 parquet 加载
            ds = load_dataset('parquet', data_dir=os.path.join(out, 'raw'))
            ds.save_to_disk(out)
            print(f'  Saved from parquet to {out}')
            return True
    except Exception as e:
        print(f'  Method2 failed: {e}')

    # 方法3: 直接下载所有文件
    try:
        files = list_repo_files(repo, repo_type='dataset')
        os.makedirs(os.path.join(out, 'raw'), exist_ok=True)
        for f in files:
            if any(f.endswith(ext) for ext in ['.json', '.jsonl', '.csv', '.parquet', '.arrow', '.txt']):
                hf_hub_download(repo, f, repo_type='dataset',
                               local_dir=os.path.join(out, 'raw'))
                print(f'  Downloaded {f}')
        print(f'  Raw files saved to {out}/raw/')
        return True
    except Exception as e:
        print(f'  Method3 failed: {e}')

    return False


# ============================================================
# 英文数据集
# ============================================================

print('=' * 50)
print('MMLU')
try_load('mmlu', 'cais/mmlu', 'all')

print('=' * 50)
print('PIQA')
try_load('piqa', 'ybisk/piqa')

print('=' * 50)
print('OpenBookQA')
try_load('openbookqa', 'allenai/openbookqa', 'main')

print('=' * 50)
print('SIQA')
try_load('siqa', 'allenai/social_i_qa')

print('=' * 50)
print('LAMBADA')
try_load('lambada', 'EleutherAI/lambada_openai', 'en')

# ============================================================
# 中文数据集
# ============================================================

print('=' * 50)
print('CMMLU (逐科目下载合并)')
cmmlu_out = os.path.join(save_dir, 'cmmlu')
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
    except Exception as e:
        # 尝试不带 trust_remote_code
        try:
            d = load_dataset('haonan-li/cmmlu', subj)
            if 'test' in d:
                all_test.append(d['test'])
            if 'dev' in d:
                all_dev.append(d['dev'])
        except:
            print(f'  skip {subj}')
print(f'  Loaded {len(all_test)}/{len(subjects)} subjects')
if all_test:
    merged = DatasetDict()
    merged['test'] = concatenate_datasets(all_test)
    if all_dev:
        merged['dev'] = concatenate_datasets(all_dev)
    for s in merged:
        print(f'  {s}: {len(merged[s])}')
    merged.save_to_disk(cmmlu_out)
    print(f'  Saved')
else:
    print('  FAILED: no subjects loaded, downloading raw files...')
    try_load('cmmlu', 'haonan-li/cmmlu')

print('=' * 50)
print('C-Eval')
try_load('ceval', 'ceval/ceval-exam', 'computer_network')  # 先测试一个

print('=' * 50)
print('C3')
try_load('c3', 'clue', 'c3')

print('=' * 50)
print('CMRC2018')
try_load('cmrc2018', 'clue', 'cmrc2018')

print('=' * 50)
print('CHID')
try_load('chid', 'clue', 'chid')

# ============================================================
# 最终汇总
# ============================================================
print('\n' + '=' * 50)
print('SUMMARY')
print('=' * 50)
for d in sorted(os.listdir(save_dir)):
    path = os.path.join(save_dir, d)
    if os.path.isdir(path):
        contents = [f for f in os.listdir(path) if f != '__pycache__']
        has_data = any(f in contents for f in ['train', 'test', 'validation', 'val', 'dev', 'raw', 'auxiliary_train'])
        status = '✅' if has_data else '❌'
        print(f'  {status} {d}: {", ".join(contents[:5])}')
