# 官谱 ↔ RPE 字段映射与单位推导

本文档记录 `pgr2rpe` 里每一条换算的依据。凡是有官方文档或参考实现可以引用的，
都给出出处；凡是只能由推理得出的，都标注"推导"并给出可验证的数字。

## 0. 三种格式的边界（先澄清命名）

| 名称 | 实际内容 | 判定依据 |
|---|---|---|
| `standard` / 官谱 | JSON：`formatVersion` + `offset` + `judgeLineList` | 无 `BPMList`，有 `judgeLineList` |
| `rpe` | JSON：`BPMList` + `META.RPEVersion` + `judgeLineList` | 有 `BPMList` |
| `pe`（PEC） | **纯文本**，`bp/cp/n1..n4` 指令 | 不以 `{` 开头 |

`pe` 不是官谱格式，也不在本文档范围内。Phigros 官谱本身没有 `pe` 这种叫法。

## 1. 单位与换算系数

### 1.1 官方定义

来源：<https://docs.lchzh.net/learning/phigros/>（"概念"一节）

```
W = 画面宽度, H = 画面高度（基准 1920×1080）
1 X  = 0.05625 W = 108 px      -> Note.positionX
1 Y  = 0.6     H = 648 px      -> Note.floorPosition、速度事件
1 T  = 1.875/BPM 秒            -> 所有时间字段，32 T = 1 拍
```

### 1.2 RPE 定义

来源：Re:PhiEdit 1.5.0 官方说明书（`_rpe_manual.txt`）与
<https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/>：

```
x ∈ [-675, 675]   (RPE_WIDTH  = 1350)
y ∈ [-450, 450]   (RPE_HEIGHT = 900), +y 向上
1 RPESpeed = 120 RPE px/s      （"速度为 1 时每秒下落 120 个 Y 单位"）
alpha ∈ [0, 255]
```

### 1.3 由此得出的系数（全部是单位定义之比，不是拟合值）

| 量 | 换算 | 系数 | 交叉验证 |
|---|---|---|---|
| 音符 x | `official_positionX × 0.05625 × 1350` | **75.9375** | Phira `parse_notes`: `position_x * (2*9/160)`，归一化后与 `1 X = 0.1125` 完全一致 |
| 判定线 x | 归一化 `[0,1]` → `(v-0.5)×1350` | 1350 | Phira `parse_move_events`: `-1 + v*2`，再 `× 2/RPE_WIDTH` |
| 判定线 y | 归一化 `[0,1]` → `(v-0.5)×900` | 900 | 同上，`× 2/RPE_HEIGHT`；`conrpepos` 的 `1-(y+450)/900` 即同一映射 |
| 速度 | `official_Y_per_s × 540 / 120` | **4.5** | Phira `SPEED_RATIO = 10/45/HEIGHT_RATIO` 与官谱侧 `/HEIGHT_RATIO` 之比恰为 4.5 |
| 旋转 | `rpe_deg = -official_deg` | −1 | Phira 对 RPE 旋转传 `factor = -1`，对官谱传 `+1` |
| 透明度 | `rpe_alpha = clamp(disappear,0,1) × 255` | 255 | Phira 对 RPE alpha 乘 `1/255`，对官谱不缩放 |
| 时间 | `tick → 秒：× 60/(32·bpm)`；`秒 → beat：× bpm/60` | — | Phira `parse_judge_line`: `r = 60/32/bpm` |
| offset | `rpe_ms = -official_s × 1000` | −1000 | 官谱"正值 = 谱面延后"，RPE"正值 = 音乐延后"，符号相反 |

> 旧工具链使用的 `positionX / 75` 与 `0.05625 W` 相差 1.25%；
> 正确值是 `1350 × 0.05625 = 75.9375`。

## 2. 官谱 → RPE 字段映射

### 2.1 根对象

| 官谱 | RPE | 处理 |
|---|---|---|
| `formatVersion` | — | 只影响移动事件解码，见 2.4 |
| `offset`（秒） | `META.offset`（整数毫秒） | `-offset × 1000`，四舍五入；亚毫秒会量化 |
| `judgeLineList` | `judgeLineList` | 一一对应 |
| — | `BPMList` | 单条 `{bpm: 主 BPM, startTime: [0,0,1]}` |
| — | `META.RPEVersion` | 默认 `170` |
| — | `chartTime` | 最后音符结束时间（秒） |
| — | `judgeLineGroup` | `["A"]` |

**主 BPM** 取"有效音符最多的那条判定线的 bpm"。官谱里 1043/1062 张谱所有线 bpm 相同；
19 张不同的谱靠"先换成秒、再换成全局 beat"处理，因此 `bpmfactor` 恒为 1。

### 2.2 判定线

| 官谱 | RPE | 说明 |
|---|---|---|
| `bpm` | `bpmfactor = 1.0` | 时间已在秒层面归一 |
| `notesAbove/notesBelow` | `notes[]` + `above` | `above = 1 / 0` |
| `speedEvents` | `eventLayers[0].speedEvents` | 常数段 → `start == end`，`×4.5` |
| `judgeLineMoveEvents` | `moveXEvents` + `moveYEvents` | 一条拆两条，时间相同 |
| `judgeLineRotateEvents` | `rotateEvents` | `× -1` |
| `judgeLineDisappearEvents` | `alphaEvents` | `× 255`，越界处切段后 clamp |
| `numOfNotes*` | `numOfNotes`（仅计数不变量） | RPE 语义：**不含 Hold** |

固定写入：`Texture: "line.png"`、`isCover: 0`、`father: -1`、`zOrder: 0`、
`anchor: [0.5, 0.5]`、空的 `*Control` 数组与 `extended` 各事件数组。

`isCover` 必须为 0：为 1 时背面音符（`above != 1`）不渲染，而官谱的 `notesBelow` 是要显示的。

### 2.3 音符

| 官谱 | RPE | 说明 |
|---|---|---|
| `type` 1/2/3/4 | 1/4/2/3 | Tap/Drag/Hold/Flick ↔ Tap/Drag/Hold/Flick，**枚举顺序不同** |
| `time` | `startTime` | tick → 秒 → beat，写成 `[m, b, d]` |
| `holdTime` | `endTime` | 仅 Hold 有意义的结束时间 |
| `positionX` | `positionX` | `× 75.9375` |
| `speed` | `speed` | 非 Hold 直接保留倍率；Hold 恒写 `1.0` |
| `floorPosition` | `yOffset` | 只写残差 `floorPosition − P_J(t)`，`×540` |
| `notesAbove/notesBelow` | `above = 1/0` | |
| — | `alpha: 255`、`size: 1.0`、`isFake: 0`、`visibleTime: 999999` | 官谱无对应概念 |

**为什么 Hold 的 `speed` 写 1.0**：官谱 Hold 的 `speed` 是"尾巴的绝对速度（Y/s）"，
实测 72,183 个 Hold 全部满足 `speed == 判定时刻线速度`（比值严格为 1.000000）；
RPE 的 `speed` 是"相对线速度的倍率"，且 Hold 头部同样被它缩放。
官谱 Hold 头部**不带**倍率，因此唯一正确的值是 1.0。
（数据核对：`ratio = note.speed / line_speed == 1.0`，样本 72,183，min = max = 1.0。）

**为什么非 Hold 的 `speed` 直接保留**：官谱非 Hold 音符的垂直距离是
`Y(t) = η · (p_N − P_J(t))`（`docs.lchzh.net/learning/phigros/calc`），
`η` 就是这里的 `speed`；RPE 的 `speed` 恰好也是同一位置的乘率，语义完全一致。
全库 4315 个非 Hold 音符的 `speed ≠ 1`，必须保留。

**为什么 `floorPosition` 只保留残差**：官谱通常满足 `p_N = P_J(t_N)`，但并非总是如此——
在 `speed = 0` 的段落里，作者会把音符"停"在离判定线很远的位置，靠时间而不是位置判定
（*Credits*、*Sparkle New Life* 的残差恒为 −100 Y）。RPE 用积分自己算落点，没有对应字段，
残差 = 音符在判定时刻偏离判定线的距离，是唯一不可由积分推出的信息。
写进 `yOffset`（`残差 × 540` px）后两种模型完全重合。实测残差 p50 = 4e-6 Y、p99 = 0.003 Y，
小于 1e-9 Y 视为浮点噪声不写。`--floor-policy drop` 可切换成"忽略 floorPosition"
（与 Phira 的处理一致）。

### 2.4 移动事件的坐标解码

| formatVersion | 编码 | 解码 |
|---|---|---|
| `1` | 左下原点，右上角 (880, 520)，字段为 `1000x + y` | `x = ⌊v/1000⌋/880`，`y = (v mod 1000)/520`，再映射到 `[-1,1]` |
| `3` | `start/end` = x，`start2/end2` = y，均为 `[0,1]` | `x = 2·v − 1` |
| 其他 | 以画面中心为原点，单位 0.1 H | `x = v·0.2·(9/16)`，`y = v·0.2`（假定 16:9） |

v1 的 `1000x+y` 在旧工具链里完全没有处理，会把 `440260` 当成归一化坐标；
本工具链解码后可直接输出 `formatVersion: 3`，顺带修掉这个缺口。

### 2.5 特殊事件：噪域 / `blockAreaList`

第九章部分谱面（如 *HATE* 各难度）在根层多出一个 `blockAreaList`：

```
[{
  topRightPercentage: {x, y},      // 归一化坐标，可超出 [0,1]
  bottomLeftPercentage: {x, y},
  appearTime, enableTime, disableTime, disappearTime,   // 单位：秒
  isSubtract: bool,                // 相加还是相减
  rotateEvents: [{anchor, time, easeType, rotation}],
  moveEvents:   [{endPosition, time, easeTypeX, easeTypeY}],
  scaleEvents:  [{anchor, time, easeTypeX, easeTypeY, scale}]
}]
```

这是"噪域"遮罩区特效：只改画面，不参与判定，RPE 没有等价机制。
处理策略是**显式忽略**：

- `official_in` 用白名单比对每一层（root / judgeLine / note / event）的键；
- 未知键收集进 `Chart.meta['ignored_fields']`，并生成一条 warning：
  `ignored unsupported field root.blockAreaList [噪域/遮罩区 (Chapter 9 block-area effect)]`；
- 转换继续，音符、判定线、时间全部照常。

注意它内部的时间单位是**秒**，与谱面其余部分的 tick 不同——如果将来要支持它，
不能直接套用 `tick → 秒` 的换算。

## 3. 事件列表的规整

官谱要求事件"首尾相接、从极小时间延伸到极大时间"（`docs.lchzh.net/learning/phigros/calc`），
但 RPE 只在事件区间内插值、区间外沿用端点值——两者都是 clamp，所以转换时：

1. 按 `startTime` 排序（全库有 70 处相邻乱序）；
2. 区间重叠时后者优先（8 处）；
3. 裁剪到 `[min(0, 首音符)−10s, 末音符+10s]`：首尾各补一段常数值，
   函数在窗口内逐点不变，窗口外的行为本来就被两种格式 clamp；
4. 合并共线相邻段（`compressed()`），只减少事件数量，不改变函数。

## 4. 校验协议

`verify.compare_charts` 对每条判定线：

1. 取两条轨道全部关键帧断点的并集，把时间轴切成小区间；
2. 在每个小区间取**中点**采样——分段线性函数只要在中点相等就在整段相等，
   这是精确判据，不是抽样近似（区间数超过 4000 时均匀抽稀，避免超长谱面拖垮批量审计）；
3. 比较 `x / y / angle / alpha(截断后) / speed`；
4. 音符按 `(时刻, 正反面, 类型, fake)` 分组、组内按 x 配对，比较
   `t / x / hold / speed / 残差`；
5. 逐音符比较**视觉落点距离**：
   - 非 Hold：官谱 `η·(ΔP + δ)` vs RPE `speed·(ΔF + δ)`；
   - Hold：头部两侧都应为 `ΔF + δ`（严格比对）；
     尾部单独统计为 `hold-body` 指标（两种模型算法不同，见 README）。

## 5. 与旧工具链的差异（供回归参考）

| 项目 | 旧链 | 本工具链 |
|---|---|---|
| `positionX` 系数 | `/75` | `/75.9375` |
| 官谱 `formatVersion=1` | 未处理 | 解码并可按 3 输出 |
| RPE `META.offset` | 读根层 `offset`（不存在） | 正确读 `META.offset` 并反转符号 |
| 官谱 alpha 越界 | 未处理 | 按穿越点切段后 clamp |
| 多 BPM 判定线 | 各自为政 | 秒级归一 + 全局 beat |
| RPE `isCover` / alpha 默认值 | 未考虑 | 显式写 `isCover=0` 与 alpha 轨道 |
| 速度换算 | `/4.5`、`/5.0`、`/10` 三种并存 | 统一 `×4.5`（有出处） |
| 噪域 `blockAreaList` | 静默丢弃 | 丢弃但显式记录，并汇入批量报告 |

## 6. PE（PEC 文本）→ RPE

PEC 是纯文本，与 JSON 无关；`pe` 与 `rpe` 都可能被存成 `.json`，所以格式只能按内容判断
（首行是一个整数毫秒值即 PEC）。权威依据：
<https://teamflos.github.io/phira-docs/chart-standard/chart-format/pe/basic.html>、
`pe/note.html`、`pe/event.html`；Phira `prpr/src/parse/pec.rs` 作为交叉验证。

### 6.1 时间轴：完全不动

PEC 与 RPE 都以 **beat** 计时，`bp <beat> <bpm>` 与 RPE `BPMList` 一一对应，
连 easing 编号都是同一套（`cm`/`cr` 的第 6/5 个参数即 RPE `easingType`）。
因此 PEC→RPE **不需要任何时间换算**，这也是它比官谱→RPE 简单得多的原因。

### 6.2 坐标

| 对象 | PEC 单位 | 换算 | 依据 |
|---|---|---|---|
| 事件 x/y（`cp`/`cm`） | 文件画布 0..2048 / 0..1400，左下原点 | `(x−1024)/1024×675`、`(y−700)/700×450` | 文档"文件坐标：中心(1024,700)，右上(2048,1400)" |
| 音符 x（`n1..n4`） | **已以 0 为中心** | `x/1024×675`（不减 1024） | 文档示例 `n1 0 0.500 -40.000 1 0`；Phira `position_x/1024` |

这是 PEC 最容易踩的坑：事件用绝对画布坐标，音符用居中坐标，两套不能混。

### 6.3 旋转

PEC **文件**里的角度是**顺时针为正**（编辑器里相反）。RPE 的旋转也是顺时针为正
（所以官谱 CCW 正值要取反、PEC 值不用）。等价链条：

```
Phira 内部旋转 R
  官谱：R = official_deg        （parse_float_events 不缩放）
  RPE ：R = -rpe_deg            （events_with_factor factor=-1）
  PEC ：R = -pec_deg            （take_f32 后取负）
=> official_deg = -pec_deg，rpe_deg = pec_deg
```

### 6.4 速度

PEC `cv` 是**瞬时设定**，Phira 除以 `5.85` 后进入与 RPE 相同的 `height` 累加器；
RPE 侧乘 `10/45/HEIGHT_RATIO`。取文档锚点"1 RPESpeed = 120 px/s（900 高画布）"
作为公共单位：

```
1 PEC 单位 = 1/5.85 屏高每秒
1 RPESpeed = 120/450 = 0.266667 屏高每秒
=> RPE = PEC × (1/5.85) / (120/450) = PEC × 0.641026
```

PEC 文档没有定义速度单位，这是全链路唯一一个"推导而非引用"的系数。它的误差只改变
音符绘制的远近视距，不改变到达时刻（时刻由 beat 决定）。

### 6.5 offset

文档："第一行为 offset，单位毫秒，计算时减去 175"。PEC 与官谱同为"正值 = 谱面延后"：

```
chart_delay = (header − 175) / 1000 s
RPE META.offset = −chart_delay × 1000 = 175 − header  (ms)
```

Phira 的 `pec.rs` 用的是 `header/1000 − 0.15`（即减 150）。差 25 ms，
`--pec-offset-bias 150` 可切换到 Phira 口径。默认按文档取 175。

### 6.6 事件展开规则

PEC 的缓动事件只给**终点值**，起点取"当前值"；瞬时事件是"立即设定"。转成 RPE 的
显式 `start/end` 时按下列规则展开，保证事件列表首尾相接、无零长度事件：

1. 事件按 `(endTime, startTime)` 排序（与 Phira `sanitize_events` 一致），
   起点早于上一个终点的重叠事件被裁到上一个终点；
2. 每个事件之前补一段"保持段" `[当前时刻, 事件起点]`，值不变——否则 RPE 会在两个
   事件之间自行插值，造成不存在的过渡；
3. 缓动事件写 `[beat0, beat1] 起值→终值 easing`；瞬时事件只更新"当前值"，
   由下一段保持段体现阶跃；
4. 末尾补一段到谱面结束，避免最后一个瞬时事件被 clamp 掉；
5. `cv` 是分段常数：每个 `cv` 展开成 `[b_i, b_{i+1}]`、`start == end` 的事件；
   首个 `cv` 晚于 0 拍时先插 `(0, 0)`（Phira 行为）。

### 6.7 不能做的事

- **PEC → 官谱**：官谱的判定线事件只有线性插值，无法表达 29 种缓动。工具链直接报错，
  不会静默压成直线。
- **PEC 负 alpha 的完整语义**：`-1` 隐藏线、`-2` 隐藏背面是 Phira 的 `pe_alpha_extension`，
  RPE 同样支持负 alpha 隐藏，但 `-1xx`/`-1xxx` 系列（提前出现时间）RPE 没有对应物，
  只能原样保留数值（RPE 会把它当作"隐藏"）。
