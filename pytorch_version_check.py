import sys
import platform
import subprocess
from datetime import datetime

def print_section(title):
    """打印分隔标题"""
    print("\n" + "="*60)
    print(title)
    print("="*60)

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

def check_python_environment():
    """检查Python环境信息"""
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"操作系统: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    
    # 检查是否为虚拟环境
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"当前在虚拟环境中: {sys.prefix}")
    else:
        print(f"当前为系统Python环境: {sys.prefix}")

def check_pytorch():
    """详细检查PyTorch安装情况"""
    try:
        # 检查PyTorch是否安装
        import torch
        print(f"\n✓ PyTorch已安装")
        print(f"  - PyTorch版本: {torch.__version__}")
        
        # 检查PyTorch编译版本
        build_info = str(torch._C._build_info())
        
        # 输出构建信息中的关键部分
        for line in build_info.split('\n'):
            if any(keyword in line.lower() for keyword in ['cuda', 'gpu', 'cudnn', 'nvidia', 'build', 'time']):
                print(f"  - {line.strip()}")
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # 是GPU版本且CUDA可用
            print(f"\n✓ PyTorch GPU版本 (CUDA可用)")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - cuDNN版本: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else '未知'}")
            
            # 获取CUDA设备信息
            device_count = torch.cuda.device_count()
            print(f"  - 可用GPU数量: {device_count}")
            
            for i in range(device_count):
                print(f"\n  GPU #{i}:")
                print(f"    - 设备名称: {torch.cuda.get_device_name(i)}")
                print(f"    - 设备能力: {torch.cuda.get_device_capability(i)}")
                
                # 获取当前设备的可用内存
                try:
                    mem_info = torch.cuda.get_device_properties(i).total_memory
                    free_mem = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                    print(f"    - 总显存: {mem_info / 1024**3:.2f} GB")
                    print(f"    - 可用显存: {free_mem / 1024**3:.2f} GB")
                except:
                    print(f"    - 无法获取内存信息")
            
            # 测试一个简单的GPU运算
            print("\n正在测试GPU计算能力...")
            try:
                x = torch.rand(1000, 1000).cuda()
                y = torch.rand(1000, 1000).cuda()
                
                start_time = datetime.now()
                z = torch.matmul(x, y)
                end_time = datetime.now()
                
                elapsed = (end_time - start_time).total_seconds() * 1000  # 毫秒
                print(f"  - 1000x1000矩阵乘法用时: {elapsed:.2f} ms")
                print(f"  ✓ GPU计算测试通过")
            except Exception as e:
                print(f"  ✗ GPU计算测试失败: {e}")
        else:
            # 检查是CPU版本还是GPU版本但无法使用CUDA
            has_cuda_functions = hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available')
            
            if has_cuda_functions:
                print(f"\n✗ PyTorch包含CUDA功能但CUDA不可用")
                print(f"  可能原因:")
                print(f"  - NVIDIA驱动未安装或版本不兼容")
                print(f"  - CUDA版本与PyTorch不兼容")
                print(f"  - GPU不支持当前的CUDA版本")
                
                # 检查NVIDIA驱动
                returncode, stdout, stderr = run_cmd('nvidia-smi')
                if returncode == 0:
                    print(f"\n  NVIDIA驱动已安装，但可能与PyTorch的CUDA版本不兼容")
                    for line in stdout.split('\n')[:3]:  # 只显示nvidia-smi输出的前几行
                        if line.strip():
                            print(f"  {line.strip()}")
                else:
                    print(f"\n  未检测到NVIDIA驱动或无法运行nvidia-smi")
            else:
                print(f"\n✓ PyTorch纯CPU版本")
                print(f"  - 不包含CUDA功能")
            
            # 测试CPU版本性能
            print("\n正在测试CPU计算能力...")
            try:
                x = torch.rand(1000, 1000)
                y = torch.rand(1000, 1000)
                
                start_time = datetime.now()
                z = torch.matmul(x, y)
                end_time = datetime.now()
                
                elapsed = (end_time - start_time).total_seconds() * 1000  # 毫秒
                print(f"  - 1000x1000矩阵乘法用时: {elapsed:.2f} ms")
                print(f"  ✓ CPU计算测试通过")
            except Exception as e:
                print(f"  ✗ CPU计算测试失败: {e}")
    
    except ImportError:
        print("\n✗ PyTorch未安装")
        print("  请使用以下命令安装PyTorch:")
        print("  - CPU版本: pip install torch torchvision torchaudio")
        print("  - GPU版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  (根据您需要的CUDA版本选择合适的命令，上面示例为CUDA 11.8)")
    
    except Exception as e:
        print(f"\n✗ 检查PyTorch时出错: {e}")

def check_cuda_toolkit():
    """检查CUDA工具包是否安装"""
    print("\n正在检测CUDA工具包...")
    
    # 检查NVIDIA驱动
    returncode, stdout, stderr = run_cmd('nvidia-smi')
    if returncode == 0:
        print("✓ NVIDIA驱动已安装")
        # 提取显卡和CUDA版本信息
        for i, line in enumerate(stdout.split('\n')):
            if i < 3 and line.strip():  # 只显示前三行
                print(f"  {line.strip()}")
    else:
        print("✗ 未检测到NVIDIA驱动")
    
    # 检查CUDA版本
    if platform.system() == "Windows":
        import os
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path:
            print(f"\n✓ 发现CUDA环境变量: {cuda_path}")
            nvcc_path = os.path.join(cuda_path, 'bin', 'nvcc.exe')
            if os.path.exists(nvcc_path):
                returncode, stdout, stderr = run_cmd(f'"{nvcc_path}" --version')
                print(f"✓ CUDA工具包已安装:")
                print(f"  {stdout.strip()}")
            else:
                print(f"✗ CUDA路径存在但未找到nvcc编译器")
        else:
            print(f"\n✗ 未设置CUDA_PATH环境变量")
    else:
        # Linux/Mac
        returncode, stdout, stderr = run_cmd('nvcc --version')
        if returncode == 0:
            print(f"\n✓ CUDA工具包已安装:")
            print(f"  {stdout.strip()}")
        else:
            print(f"\n✗ 未找到CUDA工具包或nvcc命令")

def print_recommendations(cuda_available):
    """打印建议"""
    if cuda_available:
        print("\n推荐:")
        print("1. 您的PyTorch已支持GPU加速，可以开始使用")
        print("2. 确保您的深度学习模型使用 .cuda() 或 device='cuda' 以启用GPU加速")
        print("3. 对于大模型，可使用torch.cuda.amp进行混合精度训练提高性能")
    else:
        print("\n推荐:")
        print("1. 如果您有NVIDIA显卡，建议安装支持CUDA的PyTorch版本以加速")
        print("2. 访问 https://pytorch.org/get-started/locally/ 选择适合您系统的安装命令")
        print("3. 确保您安装了与PyTorch兼容的NVIDIA驱动和CUDA版本")

def main():
    """主函数"""
    print_section("PyTorch GPU/CPU 版本检测工具")
    
    check_python_environment()
    check_cuda_toolkit()
    
    # 检查PyTorch
    pytorch_cuda_available = False
    try:
        import torch
        pytorch_cuda_available = torch.cuda.is_available()
        check_pytorch()
    except ImportError:
        print("\n✗ PyTorch未安装")
    except Exception as e:
        print(f"\n✗ 检查PyTorch时出错: {e}")
    
    print_recommendations(pytorch_cuda_available)
    
    print_section("检测完成")

if __name__ == "__main__":
    main()
    input("\n按回车键退出...") 