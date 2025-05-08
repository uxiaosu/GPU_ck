import os
import sys
import subprocess
import platform
import shutil
import winreg
from pathlib import Path
import ctypes

def is_admin():
    """检查是否具有管理员权限"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """以管理员权限重新运行脚本"""
    # 获取当前脚本的完整路径
    script = sys.argv[0]
    args = ' '.join(sys.argv[1:])
    
    # 准备以管理员权限运行的参数
    cmd = f'powershell.exe -Command "Start-Process -Verb RunAs python -ArgumentList \'{script} {args}\'"'
    
    # 执行命令
    subprocess.Popen(cmd, shell=True)
    sys.exit(0)  # 退出当前进程

def run_cmd(cmd, shell=True):
    """运行命令并返回结果"""
    try:
        process = subprocess.run(
            cmd, 
            shell=shell, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return process.returncode, process.stdout, process.stderr
    except Exception as e:
        return -1, "", str(e)

def get_installed_cuda_versions():
    """获取已安装的CUDA版本列表"""
    cuda_versions = []
    
    # 检查控制面板中已安装的程序
    try:
        print("正在查找已安装的CUDA版本...")
        # 方法1: 通过控制面板列表
        cmd = 'wmic product where "name like \'%CUDA%\'" get name,version'
        returncode, stdout, stderr = run_cmd(cmd)
        
        if returncode == 0 and stdout.strip():
            print("从控制面板中找到以下CUDA组件:")
            print(stdout)
            for line in stdout.strip().split('\n'):
                if 'CUDA' in line:
                    print(f"  - {line.strip()}")
                    cuda_versions.append(line.strip())
        else:
            print("无法通过wmic查询已安装的CUDA组件")
            
        # 方法2: 检查CUDA目录
        potential_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
            "C:\\CUDA"
        ]
        
        for base_path in potential_paths:
            if os.path.exists(base_path):
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isdir(item_path) and item.startswith("v"):
                        version = item[1:]  # 移除v前缀
                        print(f"  - 发现CUDA安装目录: {item_path} (版本 {version})")
                        cuda_versions.append(f"CUDA目录: {item_path}")
        
        # 方法3: 检查环境变量
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            print(f"  - 发现CUDA环境变量: {cuda_path}")
            cuda_versions.append(f"CUDA环境变量: {cuda_path}")
            
    except Exception as e:
        print(f"获取CUDA版本时出错: {e}")
    
    return cuda_versions

def check_gpu_details():
    """检查GPU详细信息"""
    print("\n正在检测GPU信息...")
    
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("✓ 成功检测到NVIDIA GPU:")
            # 提取关键信息
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if i < 10:  # 只显示前10行
                    print(f"  {line}")
                else:
                    print("  ...")
                    break
        else:
            print("✗ 未检测到NVIDIA GPU或驱动未正确安装")
    except FileNotFoundError:
        print("✗ 未找到nvidia-smi命令，NVIDIA驱动可能未安装")
    
    # 检查CUDA是否可用
    try:
        if platform.system() == "Windows":
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                print(f"\n✓ CUDA环境变量: {cuda_path}")
                nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
                if os.path.exists(nvcc_path):
                    result = subprocess.run([nvcc_path, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"✓ CUDA版本信息: {result.stdout.strip()}")
    except Exception as e:
        print(f"检查CUDA时出错: {e}")
    
    # 检查PyTorch是否安装及GPU是否可用
    try:
        print("\n正在检测PyTorch...")
        import torch
        print(f"✓ PyTorch已安装，版本: {torch.__version__}")
        
        # 检查CUDA是否对PyTorch可用
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"✓ PyTorch可以使用CUDA")
            print(f"  - CUDA设备数量: {device_count}")
            print(f"  - 当前CUDA设备: {current_device}")
            print(f"  - 当前设备名称: {device_name}")
            
            # 显示CUDA版本
            print(f"  - CUDA版本: {torch.version.cuda}")
        else:
            print("✗ PyTorch无法使用CUDA")
    except ImportError:
        print("✗ PyTorch未安装")
    except Exception as e:
        print(f"检查PyTorch时出错: {e}")

def uninstall_cuda():
    """卸载CUDA相关组件"""
    if platform.system() != "Windows":
        print("当前脚本仅支持Windows系统")
        return False
    
    print("\n" + "="*60)
    print("CUDA卸载工具 - 将卸载CUDA和相关组件")
    print("="*60 + "\n")
    
    # 先检测系统信息和GPU详情
    print("[1] 系统信息:")
    print(f"  - 操作系统: {platform.system()} {platform.version()}")
    print(f"  - Python版本: {platform.python_version()}")
    
    print("\n[2] GPU和CUDA详细信息:")
    check_gpu_details()
    
    # 获取已安装的CUDA组件
    print("\n[3] 已安装的CUDA组件:")
    cuda_versions = get_installed_cuda_versions()
    
    if not cuda_versions:
        print("\n未检测到已安装的CUDA组件")
        input("\n按回车键退出...")
        return False
    
    # 提供选项菜单
    while True:
        print("\n" + "="*60)
        print("请选择操作:")
        print("1. 卸载所有CUDA组件")
        print("2. 退出")
        print("="*60)
        
        choice = input("\n请输入选项 (1-2): ")
        
        if choice == "2":
            print("操作已取消")
            return False
        elif choice == "1":
            break
        else:
            print("无效选项，请重新输入")
    
    print("\n是否确定要卸载以上列出的所有CUDA组件?")
    print("此操作不可逆，将会卸载NVIDIA CUDA工具包及其相关组件")
    confirm = input("输入'yes'确认卸载: ")
    
    if confirm.lower() != 'yes':
        print("取消卸载操作")
        return False
    
    print("\n开始卸载CUDA组件...\n")
    
    # 1. 首先卸载通过控制面板安装的CUDA组件
    uninstall_commands = [
        'wmic product where "name like \'%CUDA%\'" call uninstall /nointeractive',
        'wmic product where "name like \'%NVIDIA%\'" call uninstall /nointeractive'
    ]
    
    for cmd in uninstall_commands:
        print(f"执行命令: {cmd}")
        returncode, stdout, stderr = run_cmd(cmd)
        if returncode == 0:
            print("  命令执行成功")
        else:
            print(f"  命令执行失败，错误信息: {stderr}")
    
    # 2. 删除CUDA目录
    cuda_paths = [
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\CUDA"
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"删除CUDA目录: {path}")
            try:
                shutil.rmtree(path)
                print(f"  成功删除目录: {path}")
            except Exception as e:
                print(f"  无法删除目录: {path}，错误: {e}")
    
    # 3. 清理环境变量
    try:
        print("\n清理CUDA相关环境变量...")
        env_vars_to_remove = ['CUDA_PATH', 'CUDA_HOME']
        
        for var in env_vars_to_remove:
            if var in os.environ:
                print(f"  移除环境变量: {var}")
                cmd = f'setx {var} "" /M'
                run_cmd(cmd)
        
        # 从PATH中移除CUDA路径
        path_var = os.environ.get('PATH', '')
        new_paths = []
        for p in path_var.split(';'):
            if 'cuda' not in p.lower() and 'nvidia' not in p.lower():
                new_paths.append(p)
            else:
                print(f"  从PATH中移除: {p}")
        
        if len(new_paths) < len(path_var.split(';')):
            new_path = ';'.join(new_paths)
            cmd = f'setx PATH "{new_path}" /M'
            run_cmd(cmd)
    except Exception as e:
        print(f"清理环境变量时出错: {e}")
    
    print("\n" + "="*60)
    print("CUDA卸载操作已完成")
    print("建议重启计算机以完成清理过程")
    print("="*60)
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUDA检测与卸载工具")
    print("="*60)
    
    # 检查管理员权限，如果没有则重新以管理员权限运行
    if not is_admin():
        print("脚本需要管理员权限才能完全卸载CUDA组件")
        print("正在请求管理员权限...")
        run_as_admin()
    else:
        # 已获得管理员权限，执行卸载操作
        uninstall_cuda()
        input("\n按回车键退出...") 