from pathlib import Path
from safe_subprocess import run

LANG_NAME = "Cangjie"
LANG_EXT = ".cj"

def eval_script(path: Path):
    # 获取不带扩展名的文件名作为输出文件名
    basename = ".".join(str(path).split(".")[:-1])
    
    # 编译阶段
    build_result = run(["cjc", str(path), "-o", basename])
    if build_result.exit_code != 0:
        return {
            "status": "SyntaxError",
            "exit_code": build_result.exit_code,
            "stdout": build_result.stdout,
            "stderr": build_result.stderr,
        }

    # 运行阶段
    run_result = run([basename])
    
    # 清理生成的可执行文件
    if Path(basename).exists():
        Path(basename).unlink()
    
    # 处理运行结果
    if run_result.timeout:
        status = "Timeout"
    elif run_result.exit_code == 0:
        status = "OK"
    else:
        status = "Exception"
        
    return {
        "status": status,
        "exit_code": run_result.exit_code,
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
    }

if __name__ == "__main__":
    from generic_eval import main
    main(eval_script, LANG_NAME, LANG_EXT)