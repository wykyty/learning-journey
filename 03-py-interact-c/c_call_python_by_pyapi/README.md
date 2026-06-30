
## c_call_python_by_api
### 编译
shell编译命令
编译失败：
```
gcc main.c -std=c99 $(pkg-config --cflags --libs python3) -fno-lto -o main
/usr/bin/ld: /tmp/ccEHvhOz.o: in function `main':
main.c:(.text+0x10): undefined reference to `Py_Initialize'
/usr/bin/ld: main.c:(.text+0x1f): undefined reference to `PyRun_SimpleStringFlags'
/usr/bin/ld: main.c:(.text+0x2e): undefined reference to `PyRun_SimpleStringFlags'
/usr/bin/ld: main.c:(.text+0x3d): undefined reference to `PyRun_SimpleStringFlags'
/usr/bin/ld: main.c:(.text+0x47): undefined reference to `PyImport_ImportModule'
/usr/bin/ld: main.c:(.text+0x66): undefined reference to `Py_Finalize'
/usr/bin/ld: main.c:(.text+0x81): undefined reference to `PyObject_GetAttrString'
/usr/bin/ld: main.c:(.text+0x98): undefined reference to `PyCallable_Check'
/usr/bin/ld: main.c:(.text+0xb0): undefined reference to `Py_Finalize'
/usr/bin/ld: main.c:(.text+0xc4): undefined reference to `PyTuple_New'
/usr/bin/ld: main.c:(.text+0xe6): undefined reference to `Py_BuildValue'
/usr/bin/ld: main.c:(.text+0xfa): undefined reference to `PyTuple_SetItem'
/usr/bin/ld: main.c:(.text+0x10d): undefined reference to `PyObject_CallObject'
/usr/bin/ld: main.c:(.text+0x11d): undefined reference to `PyLong_AsLong'
/usr/bin/ld: main.c:(.text+0x13c): undefined reference to `Py_Finalize'
collect2: error: ld returned 1 exit status
```
编译失败的根本原因是链接器（Linker）找不到 Python 的动态链接库。报错信息中大量的 undefined reference to ... 说明编译器找到了头文件，但在链接阶段未能成功关联 Python 的底层 C API 库。

换命令：
```
gcc main.c -std=c99 -fno-lto -o main $(python3-config --cflags --ldflags --embed)
```
编译成功。

### 运行
```
./main
```
报错：
```
./main
./main: error while loading shared libraries: libpython3.10.so.1.0: cannot open shared object file: No such file or directory
```
如果编译成功，但在运行 `./main` 时提示 `loading shared libraries: libpython3.x.so: cannot open shared object file`，说明系统找不到 Python 的动态链接库。

**解决方法**：
在 `/etc/ld.so.conf.d/` 目录下新建一个配置文件（如 `python3.conf`），将 Python 的库路径写入其中，然后执行 `ldconfig` 刷新缓存：
```bash
# 1. 创建配置文件并写入路径（以 Python 3.10 为例，请根据实际路径调整）
echo "/usr/lib/x86_64-linux-gnu" | sudo tee /etc/ld.so.conf.d/python3.conf
# 2. 刷新动态链接库缓存
sudo ldconfig
```

配置了路径但仍然报错，通常是因为**路径配置不准确**或者**缺少对应的符号链接**。我们可以按照以下步骤进行排查和彻底解决：

### 第一步：精准定位库文件的实际路径
首先，我们需要确认系统中 `libpython3` 相关的库文件究竟存放在哪里。请在终端执行以下全盘搜索命令：
```bash
find / -name "libpython3*.so*" 2>/dev/null
```
*(注意：如果提示权限不足，请在前面加上 `sudo`)*

### 第二步：根据搜索结果对症下药

**情况 1：搜索到了具体的库文件路径（例如 `/usr/local/lib/libpython3.10.so.1.0`）**
这说明库存在，但系统链接器未能正确识别。你可以采用以下两种方法之一：
*   **方法 A（推荐，创建符号链接）**：直接将找到的库文件链接到系统的默认库目录 `/usr/lib/` 下，然后刷新缓存：
    ```bash
    sudo ln -s /usr/local/lib/libpython3.10.so.1.0 /usr/lib/
    sudo ldconfig
    ```
*   **方法 B（更新环境变量）**：将找到的库所在目录（如 `/usr/local/lib`）添加到环境变量中：
    ```bash
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    ```
    *(如果希望永久生效，可将该命令追加到 `~/.bashrc` 文件中并执行 `source ~/.bashrc`)*

这里采用方法A，创建符号链接，成功输出：
```
./main
hello world
py_list_sum([5, 6, 7]) = 18 
```










# reference

https://github.com/thb1314/python_interact_c

