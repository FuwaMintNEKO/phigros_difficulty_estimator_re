# phigros_difficulty_estimator_re

Phigros 谱面定数估算器。输入一张谱面（官谱 / RPE / PEC / 谱包），输出**官谱尺度**的定数，
并把定数拆成五类玩家能看懂的构成，不区分 EZ / HD / IN / AT。

两部分：

- `pgr2rpe/`：谱面解析与格式转换（官谱 ⇄ RPE，PEC → RPE），带两套独立实现互证；
- `rating/`：特征提取、定数模型、批量预测；`app.py` 是本地网页界面。

## 快速开始

```bash
pip install -r requirements.txt
python app.py                       # http://127.0.0.1:5000，拖入谱面即可，可多选
```

界面里每张谱面一张卡片：定数（0–18 绿到红渐变）、社区尺度换算、谱面自带的信息
（时长、BPM 范围、曲师、谱师）、Tap / Hold / Drag / Flick 占比环图、五类难度构成，
以及这张谱面自己的变速滑杆。可按定数升降序排列、单张移除或清空。

批量预测：

```bash
python -m rating.batch <目录或文件...> --out result.csv [--speed 1.3]
```

支持 `.json` / `.pec` / `.pez` / `.zip` 和 `*.Charter.0` 谱包文件夹（会读取 `info.json`）。
输出每张谱面的定数、五类占比、格式、路径。

## 它怎么估

1. **解析**：统一成「秒 / 归一化 x / 屏高每秒」。忽略 `isFake` 音符；`holdTime ≤ 0` 的 Hold 当 Tap；
   JSON `null` 字段按默认值；停在 1e9 tick 的哨兵音符会被裁掉，不拉长时长。
2. **特征**：106 维，全部是玩家能理解的量，用连续核和分位数，不设人工阈值。
   密度、多押、尾杀只统计 **Tap + Hold**（Drag / Flick 不需要按压，堆叠不代表难）。
3. **模型**：GBR + 残差 Ridge（输出对特征连续），官谱模型与高手表模型按得分平滑混合，
   再经单调校准、分档偏差校正，以及 17 以上的小幅平滑提升。
4. **外推**：超出训练包络的谱面（超高密度、4K 等）朝训练集最难标签有界收敛，不会无限外推。
5. **难度构成**：把某一类特征换成训练集中位谱，重新预测，差值归一化为百分比。
6. **变速**：按 Phira 语义整谱时间压缩（`TimeManager::now() = real * speed`），
   所有时刻 ÷k、流速 ×k 后重新提特征。多押是"50 ms 窗口内整数计数"，原始曲线会有台阶，
   界面显示的是采样后取单调包络的结果。

## 精度（自制谱留出，47 张，社区尺度）

| 指标 | 数值 |
|---|---|
| 官谱 5 折（按歌曲分组）MAE | 0.69 |
| 17+自制谱表留出 MAE / bias | 0.39 / −0.11 |
| 17–18 段 / 18–18.5 段 / 18.5+ 段 MAE | 0.39 / 0.31 / 0.50 |

18.5 以上样本少、标注本身偏高，这一段预测偏低属于已知情况。`models/model.pkl` 是 pickle，
**只加载你信任的文件**；它依赖 `requirements.txt` 里固定的 scikit-learn 版本。

## 训练（可选）

训练数据不随仓库发布（谱面不是我的，不便再分发）。需要自己重训时：

```bash
export PHIGROS_DATA_DIR=/path/to/data     # 目录结构见 rating/dataset.py 顶部注释
python -m rating.build && python -m rating.train
```

## 测试

```bash
python tests/test_equivalence.py     # 17 项：官谱 <-> RPE
python tests/test_pec.py             # 13 项：PEC -> RPE（含与独立实现的逐点对比）
python tests/test_rating.py          # 解析与特征的正确性，使用合成谱，克隆即可跑
```

格式转换的字段映射与单位推导见 [`docs/FORMAT_MAPPING.md`](docs/FORMAT_MAPPING.md)。

## 许可

MIT，见 [`LICENSE`](LICENSE)。

## 上传者备注
感谢deepseek-v4-flash-0731,deepseek-v4-pro-0813,deepseek-v4.1-flash,deepseek-v4.1-flash-1017,gpt-6.1-sol,gpt-6.1-astra,gemini-3.8-flash,claude-sonnet-5-5,grok-4.7的开发
本次是基于Phigros4.0.1.2版本开发的定数预测系统，难度尺度用18.0对应社区的18.8，代码里有很多屎山，没有屎山的版本我尝试过了，效果非常一般，我也不知道怎么做出来的，但是鉴于黑盒测试效果还不错，屎山就留在这儿了，期待更多大佬做出更好的预测系统。
never say 「give up」.
