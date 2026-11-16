# 10年期国债收益率研究仪表盘

本项目是一个本地运行的国债收益率研究面板，包含以下能力：

- 读取本地 Excel 数据并展示历史走势
- 运行模拟预测与多变量分析
- 通过本地新闻代理获取债市新闻参考
- 调用 DeepSeek 生成 AI 分析与新闻解读

## 目录说明

- `html_demo/bond_yield_dashboard.html`：主页面入口
- `data/10_bond_rate_ori.xlsx`：默认收益率数据文件
- `data/other_data.xlsx`：相关参考数据文件
- `news_proxy.py`：本地新闻代理服务
- `start_dashboard.bat`：一键启动脚本
- `token_api.config`：DeepSeek API Key 配置文件

## 环境要求

- Windows
- Python 3.7 及以上
- Chrome / Edge 等现代浏览器

## 需要安装哪些 Python 包

当前版本的本地后端 `news_proxy.py` 只使用 Python 标准库实现：

- `http.server`
- `urllib`
- `json`
- `socketserver`
- `datetime`

因此：

- **不需要额外执行 `pip install`**
- 只要本机能运行 `python` 或 `py` 命令即可

## API Key 配置

### 1. DeepSeek Key

页面会优先尝试读取：

```text
token_api.config
```

当前文件内容示例：

```text
sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

说明：

- 文件中直接写 API Key 即可
- 页面会通过 `../token_api.config` 自动读取
- 如果读取失败，页面会弹窗提示临时输入（版本库中不内置真实密钥）

### 2. 新闻搜索 Key

新闻搜索通过本地 `news_proxy.py` 调用智谱 MCP。

出于安全考虑，版本库中不包含搜索 Key。`news_proxy.py` 按以下优先级读取密钥：

1. 环境变量 `BIGMODEL_API_KEY`
2. 本地未入库的 `search_api.config` 文件（内容只写一行 Key 即可）

环境变量方式示例（启动前设置）：

```powershell
$env:BIGMODEL_API_KEY="你的智谱搜索Key"
```

## 如何启动

### 方式一：推荐，双击一键启动

直接双击：

```text
start_dashboard.bat
```

该脚本会自动完成：

1. 启动本地静态文件服务，端口 `8000`
2. 启动本地新闻代理服务，端口 `8001`
3. 自动打开浏览器页面

### 方式二：手动启动

在项目根目录执行：

```powershell
python -m http.server 8000 --bind 0.0.0.0
```

再打开另一个终端执行：

```powershell
python news_proxy.py
```

如果本机没有 `python` 命令，也可以尝试：

```powershell
py -m http.server 8000 --bind 0.0.0.0
py news_proxy.py
```

## 打开什么网页

启动成功后，浏览器访问：

```text
http://localhost:8000/html_demo/bond_yield_dashboard.html
```

新闻代理健康检查地址：

```text
http://localhost:8001/api/health
```

正常返回示例：

```json
{"ok": true, "service": "news-proxy"}
```

### 局域网内其他人如何访问

如果其他人与当前电脑处于同一个局域网，可通过当前电脑的局域网 IP 访问页面。

步骤如下：

1. 在当前电脑执行 `ipconfig`
2. 找到本机 IPv4 地址，例如 `192.168.1.23`
3. 其他人在同一局域网浏览器中打开：

```text
http://192.168.1.23:8000/html_demo/bond_yield_dashboard.html
```

说明：

- 当前版本的静态服务已绑定 `0.0.0.0`
- 当前版本的新闻代理也已绑定 `0.0.0.0`
- 前端会自动根据访问主机名拼接新闻代理地址，因此局域网访问时新闻模块也可正常请求
- 若仍无法访问，请检查 Windows 防火墙是否拦截 `8000` 和 `8001` 端口

## 使用步骤

1. 启动 `start_dashboard.bat`
2. 打开页面 `http://localhost:8000/html_demo/bond_yield_dashboard.html`
3. 确认默认 Excel 自动加载成功
4. 点击“刷新新闻参考”获取近期新闻
5. 点击“解读新闻”生成新闻总结
6. 点击“生成 AI 分析”生成收益率与新闻的综合分析

## 端口说明

- `8000`：静态页面服务
- `8001`：新闻代理服务

不要把 `news_proxy.py` 的端口改成 `8000`，否则会与静态页面服务冲突。

## 常见问题

### 1. 双击 HTML 打开后 Excel 无法自动加载

原因：

- 浏览器会阻止 `file://` 页面直接读取本地文件

解决：

- 不要直接双击 `bond_yield_dashboard.html`
- 请通过 `start_dashboard.bat` 或 `python -m http.server` 启动后访问

### 2. 页面能打开，但新闻搜索失败

先检查：

```text
http://localhost:8001/api/health
```

如果打不开，说明 `news_proxy.py` 没有正常启动。

如果是局域网内其他人访问，请改为检查：

```text
http://你的局域网IP:8001/api/health
```

### 3. 修改了 `news_proxy.py`，但页面效果没变化

原因：

- 很可能 `8001` 上跑的还是旧进程

解决：

- 关闭旧的新闻代理终端
- 重新运行 `start_dashboard.bat`
- 或重新执行：

```powershell
python news_proxy.py
```

### 4. API Key 读取失败

检查：

- `token_api.config` 是否存在
- 文件里是否只写了一行 API Key
- 页面是否通过 `http://localhost:8000/...` 打开

## 备注

- 当前新闻搜索优先走定制站点检索，再做本地强过滤，全网 MCP 搜索只作为补充
- 历史走势图与模拟测试图表已支持鼠标悬浮显示数据

## 2026-11-12 优化：接入真实预测模型（40/41 脚本管线）

本次升级把面板的"模拟预测"升级为可直接查看生产管线的真实信号：

1. **新增"真实模型信号"模块**（导航栏第 5 项）
   - 四周期（h5/h10/h15/h20）方向预测表：XGB / BiGRU NN / 融合 / LLM 修正
     概率、到期日、三组隐含点位
   - 下行概率对比条（XGB/融合/LLM修正 三色）
   - 预测台账（最近 12 条历史信号）与口径说明
   - 数据源：预测项目 `data_new/预测记录.csv` 与 `data_new/out/大模型修正.csv`

2. **news_proxy.py 新增 `GET /api/signal`**
   - 读取预测台账与 LLM 修正并标准化为 JSON，供面板拉取
   - 项目根目录可用环境变量 `BOND_PROJECT_DIR` 覆盖（默认同级
     `../43_华泰证券预处理`）

3. **新闻搜索免费源回退**
   - 智谱 MCP `web_search_prime` 未配置 key / 调用失败时，自动回退到
     新浪财经滚动 + 东方财富 7x24 快讯的关键词直采，接口不再直接 500
   - 命中条目标注 `retrievalLabel: 免费源直采（新浪/东财快讯回退）`

4. **start_dashboard.bat Python 探测修复**
   - 优先使用 `D:\soft\Anaconda3\python.exe`，避免 Windows 商店 python
     stub 导致启动失败；其次 py 启动器，最后 python

### 使用

```text
1. 先在预测项目运行: python model_new/40_最新时点预测.py（与可选的 41_大模型信息修正.py）
2. 双击 start_dashboard.bat
3. 打开导航"真实模型信号"
```
