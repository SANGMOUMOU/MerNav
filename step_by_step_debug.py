from dotenv import load_dotenv
load_dotenv()

import faulthandler
faulthandler.enable()

import sys
sys.path.insert(0, 'src')

print("步骤 1: 导入模块")
import yaml
from WMNav_env import WMNavEnv
print("  ✅ 导入成功")

print("\n步骤 2: 加载配置")
with open('config/WMNav.yaml', 'r') as file:
    config = yaml.safe_load(file)
config['env_cfg']['dataset'] = 'hm3d_v0.2'
print("  ✅ 配置加载成功")

print("\n步骤 3: 创建环境 (调用 __init__)")
print("  3.1 初始化logging...")
env = None
try:
    # 手动逐步初始化
    import os
    from src.WMNav_env import WMNavEnv
    
    # 先不创建完整对象，手动测试
    print("  3.2 测试数据集路径...")
    dataset_root = os.environ.get("DATASET_ROOT")
    objnav_path = "objectnav_hm3d_v0.2"
    split = config['env_cfg']['split']
    
    content_path = os.path.join(dataset_root, objnav_path, f'{split}/content')
    print(f"      路径: {content_path}")
    print(f"      存在: {os.path.exists(content_path)}")
    
    if os.path.exists(content_path):
        files = os.listdir(content_path)
        print(f"      文件数: {len(files)}")
        print(f"      前3个: {files[:3]}")
    
    print("\n  3.3 现在创建完整环境对象...")
    env = WMNavEnv(cfg=config)
    print("  ✅ 环境创建成功")
    
except Exception as e:
    print(f"  ❌ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n步骤 4: 环境创建成功，检查属性")
print(f"  - num_episodes: {len(env.all_episodes)}")
print(f"  - agent类型: {type(env.agent)}")

print("\n步骤 5: 准备运行实验")
print("  (如果这之后崩溃，问题在run_experiment中)")

print("\n=== 所有初始化步骤完成 ===")
