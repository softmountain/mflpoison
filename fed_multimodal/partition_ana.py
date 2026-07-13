"""
诊断当前实验使用的数据划分配置
检查partition文件和特征文件的一致性
"""
import os
import json
from pathlib import Path
from collections import defaultdict

# 配置路径
base_dir = Path(__file__).parent
output_dir = base_dir / "results"
dataset = "uci-har"
alpha = 5.0
alpha_str = str(alpha).replace('.', '')

print("=" * 80)
print("UCI-HAR数据划分诊断报告")
print("=" * 80)

# 1. 检查partition文件
print("\n[步骤1] 检查Partition文件")
print("-" * 80)

partition_path = output_dir / "partition" / dataset / f"partition_alpha{alpha_str}.json"
partition_exists = partition_path.exists()

if partition_exists:
    print(f"✓ 文件存在: {partition_path}")
    
    with open(partition_path, 'r') as f:
        partition = json.load(f)
    
    train_clients = [k for k in partition.keys() if k not in ['dev', 'test']]
    
    # 分析分片数
    user_shards = defaultdict(list)
    for cid in train_clients:
        if '-' in cid:
            user_id, shard_id = cid.split('-')
            user_shards[user_id].append(int(shard_id))
    
    max_shards = max(len(shards) for shards in user_shards.values()) if user_shards else 0
    all_shard_ids = sorted(set(sid for shards in user_shards.values() for sid in shards))
    
    print(f"  训练客户端总数: {len(train_clients)}")
    print(f"  唯一用户数: {len(user_shards)}")
    print(f"  每用户最大分片数: {max_shards}")
    print(f"  分片ID范围: {all_shard_ids}")
    print(f"  推断的num_clients: {max_shards}")
    print(f"  预期客户端数 (21用户): {21 * max_shards}")
    
    # 显示示例客户端
    sample_user = sorted(user_shards.keys())[0]
    sample_clients = [f"{sample_user}-{sid}" for sid in sorted(user_shards[sample_user])]
    print(f"  示例客户端（用户{sample_user}）: {sample_clients[:5]}{'...' if len(sample_clients) > 5 else ''}")
    
    partition_num_clients = max_shards
    partition_client_count = len(train_clients)
else:
    print(f"✗ 文件不存在: {partition_path}")
    print(f"  请先运行: python data_partition.py --alpha {alpha} --num_clients <N>")
    partition_num_clients = None
    partition_client_count = 0

# 2. 检查特征文件
print("\n[步骤2] 检查特征文件")
print("-" * 80)

feat_acc_path = output_dir / "feature" / "acc" / dataset / f"alpha{alpha_str}"
feat_gyro_path = output_dir / "feature" / "gyro" / dataset / f"alpha{alpha_str}"

acc_exists = feat_acc_path.exists()
gyro_exists = feat_gyro_path.exists()

if acc_exists:
    print(f"✓ ACC特征目录存在: {feat_acc_path}")
    acc_files = [f for f in os.listdir(feat_acc_path) if f.endswith('.pkl')]
    acc_clients = [f.split('.pkl')[0] for f in acc_files]
    acc_train_clients = [c for c in acc_clients if c not in ['dev', 'test']]
    
    # 分析分片
    acc_shard_ids = set()
    for cid in acc_train_clients:
        if '-' in cid:
            shard_id = int(cid.split('-')[1])
            acc_shard_ids.add(shard_id)
    
    print(f"  .pkl文件总数: {len(acc_files)}")
    print(f"  训练客户端数: {len(acc_train_clients)}")
    print(f"  分片ID范围: {sorted(acc_shard_ids)}")
    print(f"  推断的num_clients: {max(acc_shard_ids) + 1 if acc_shard_ids else 0}")
    
    acc_num_clients = max(acc_shard_ids) + 1 if acc_shard_ids else 0
else:
    print(f"✗ ACC特征目录不存在: {feat_acc_path}")
    print(f"  请先运行: python extract_feature.py --alpha {alpha} --dataset {dataset}")
    acc_num_clients = None
    acc_train_clients = []

if gyro_exists:
    print(f"✓ GYRO特征目录存在: {feat_gyro_path}")
    gyro_files = [f for f in os.listdir(feat_gyro_path) if f.endswith('.pkl')]
    print(f"  .pkl文件总数: {len(gyro_files)}")
else:
    print(f"✗ GYRO特征目录不存在: {feat_gyro_path}")

# 3. 一致性检查
print("\n[步骤3] 一致性检查")
print("-" * 80)

if partition_exists and acc_exists:
    # 检查数量
    if partition_client_count == len(acc_train_clients):
        print(f"✓ 客户端数量一致: {partition_client_count}")
    else:
        print(f"✗ 客户端数量不一致！")
        print(f"    Partition文件: {partition_client_count} 个训练客户端")
        print(f"    ACC特征文件: {len(acc_train_clients)} 个训练客户端")
        print(f"  可能原因：")
        print(f"    1. 重新运行了data_partition.py但未重新运行extract_feature.py")
        print(f"    2. 特征提取不完整")
    
    # 检查num_clients
    if partition_num_clients == acc_num_clients:
        print(f"✓ num_clients一致: {partition_num_clients}")
    else:
        print(f"✗ num_clients不一致！")
        print(f"    Partition推断: num_clients={partition_num_clients}")
        print(f"    特征文件推断: num_clients={acc_num_clients}")
        print(f"  这是严重问题！数据不匹配！")
    
    # 检查客户端名称
    partition_set = set(train_clients)
    feature_set = set(acc_train_clients)
    
    missing_in_feature = partition_set - feature_set
    extra_in_feature = feature_set - partition_set
    
    if not missing_in_feature and not extra_in_feature:
        print(f"✓ 客户端名称完全匹配")
    else:
        if missing_in_feature:
            print(f"✗ Partition中有但特征中缺失的客户端: {len(missing_in_feature)}个")
            print(f"    示例: {list(missing_in_feature)[:5]}")
        if extra_in_feature:
            print(f"✗ 特征中有但Partition中不存在的客户端: {len(extra_in_feature)}个")
            print(f"    示例: {list(extra_in_feature)[:5]}")
            print(f"  可能是旧数据残留，建议清理！")
    
    # 检查GYRO与ACC一致性
    if gyro_exists:
        acc_set = set(acc_files)
        gyro_set = set(gyro_files)
        if acc_set == gyro_set:
            print(f"✓ ACC和GYRO特征文件完全一致")
        else:
            print(f"✗ ACC和GYRO特征文件不一致！")

elif not partition_exists:
    print(f"⚠ 无法进行一致性检查：Partition文件不存在")
elif not acc_exists:
    print(f"⚠ 无法进行一致性检查：特征文件不存在")

# 4. 训练配置预测
print("\n[步骤4] 训练配置预测")
print("-" * 80)

if acc_exists:
    print(f"当您运行 train.py --alpha {alpha} 时：")
    print(f"  程序将读取: {feat_acc_path}")
    print(f"  检测到的客户端数: {len(acc_train_clients)}")
    print(f"  如果sample_rate=0.1，每轮采样: {int(len(acc_train_clients) * 0.1)} 个客户端")
    
    # 预测攻击影响
    attacked_clients = [c for c in acc_train_clients if c.endswith('-1')]
    if attacked_clients:
        print(f"\n  标签翻转攻击分析（target_client_suffix='-1'）:")
        print(f"    被攻击的客户端数: {len(attacked_clients)}")
        print(f"    攻击比例: {len(attacked_clients)/len(acc_train_clients)*100:.1f}%")
        print(f"    每轮期望恶意客户端: {len(attacked_clients) * 0.1:.2f} 个")
else:
    print(f"无法预测训练配置：特征文件不存在")

# 5. 建议
print("\n[步骤5] 建议操作")
print("-" * 80)

if not partition_exists:
    print("1. 运行数据划分:")
    print(f"   python data_partition.py --alpha {alpha} --num_clients 10")
elif not acc_exists:
    print("1. 运行特征提取:")
    print(f"   python extract_feature.py --alpha {alpha} --dataset {dataset}")
elif partition_exists and acc_exists:
    if partition_client_count == len(acc_train_clients) and partition_num_clients == acc_num_clients:
        print("✓ 所有检查通过，可以开始训练！")
        print(f"\n  运行训练:")
        print(f"   python train.py --alpha {alpha} --dataset {dataset}")
    else:
        print("⚠ 发现不一致，建议操作:")
        print(f"  1. 删除旧的特征文件:")
        print(f"     rm -rf {feat_acc_path}")
        print(f"     rm -rf {feat_gyro_path}")
        print(f"  2. 重新运行特征提取:")
        print(f"     python extract_feature.py --alpha {alpha} --dataset {dataset}")
        print(f"  3. 重新运行此诊断脚本确认")

# 6. 文件修改建议
print("\n[步骤6] 避免未来冲突的建议")
print("-" * 80)
print("当前系统的问题：")
print("  1. partition文件名不包含num_clients，容易覆盖")
print("  2. 特征文件路径不包含num_clients，容易混淆")
print("\n建议修改代码：")
print("  1. 修改partition文件名为: partition_alpha{alpha}_nc{num_clients}.json")
print("  2. 修改特征路径为: feature/acc/{dataset}/alpha{alpha}_nc{num_clients}/")
print("  3. 在train.py中添加--num_clients参数明确指定使用的划分")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
