import os
import sys
import subprocess
import platform

def check_nvidia_smi():
    """检查是否能运行nvidia-smi命令"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi 命令执行成功，发现NVIDIA显卡")
            print(result.stdout.split('\n')[0])  # 打印第一行，通常包含驱动版本信息
            return True
        else:
            print("✗ nvidia-smi 命令执行失败，未找到NVIDIA显卡或驱动未正确安装")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi 命令不存在，未找到NVIDIA驱动")
        return False

def check_cuda():
    """检查CUDA是否可用及版本"""
    try:
        if platform.system() == "Windows":
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                print(f"✓ 发现CUDA环境变量: {cuda_path}")
                nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
                if os.path.exists(nvcc_path):
                    result = subprocess.run([nvcc_path, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    print(f"✓ CUDA版本信息: {result.stdout.strip()}")
                    return True
                else:
                    print(f"✗ 未找到NVCC编译器: {nvcc_path}")
            else:
                print("✗ 未设置CUDA_PATH环境变量")
        else:
            # Linux/Mac
            try:
                result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print(f"✓ CUDA版本信息: {result.stdout.strip()}")
                    return True
                else:
                    print("✗ NVCC命令执行失败")
                    return False
            except FileNotFoundError:
                print("✗ 未找到NVCC命令，CUDA可能未安装")
                return False
    except Exception as e:
        print(f"✗ 检查CUDA时出错: {e}")
    return False

def check_pytorch():
    """检查PyTorch是否安装及GPU是否可用"""
    try:
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
            
            # 测试张量是否能成功移至GPU
            try:
                x = torch.tensor([1.0])
                x = x.cuda()
                print("✓ 成功创建GPU张量")
            except Exception as e:
                print(f"✗ 无法创建GPU张量: {e}")
        else:
            print("✗ PyTorch无法使用CUDA")
            
        return cuda_available
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    except Exception as e:
        print(f"✗ 检查PyTorch时出错: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "="*50)
    print("GPU检测工具 - 检测GPU、CUDA和PyTorch")
    print("="*50 + "\n")
    
    print("[1] 系统信息:")
    print(f"  - 操作系统: {platform.system()} {platform.version()}")
    print(f"  - Python版本: {platform.python_version()}")
    print("\n")
    
    print("[2] 检测NVIDIA GPU:")
    nvidia_gpu = check_nvidia_smi()
    print("\n")
    
    print("[3] 检测CUDA安装:")
    cuda_installed = check_cuda()
    print("\n")
    
    print("[4] 检测PyTorch CUDA支持:")
    pytorch_cuda = check_pytorch()
    print("\n")
    
    print("="*50)
    print("总结:")
    print(f"  NVIDIA GPU: {'可用' if nvidia_gpu else '不可用'}")
    print(f"  CUDA安装: {'已安装' if cuda_installed else '未安装或不可用'}")
    print(f"  PyTorch CUDA支持: {'支持' if pytorch_cuda else '不支持'}")
    print("="*50)

if __name__ == "__main__":
    main() 