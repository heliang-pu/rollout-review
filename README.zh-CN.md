# rollout-review

[English](README.md) | **中文**

在浏览器里一集一集地看 [LeRobot](https://github.com/huggingface/lerobot) v2.1 数据集的回放，手工给出结论。
用来在训练前挑出脏数据：画面里有手、物体中途被碰、标签不对。

- 每一集的所有相机拼成一个画面，左上角是数据集名、集号和任务，右上角是帧号。
- 结论三选一：没问题、有问题、不确定；可以加标签和备注。全程可以只用键盘。
- 只读数据集，什么都不往数据集里写。

只用 Python 标准库（3.10 以上），不用安装。拼视频需要 `ffmpeg` 和 `ffprobe`。

## 用法

```bash
git clone https://github.com/heliang-pu/rollout-review && cd rollout-review

./rollout-review serve data/real_pick data/real_pour          # 打开 http://127.0.0.1:8790/
./rollout-review serve data/real --host 0.0.0.0 --port 8790   # 让同一网络里的平板也能访问
./rollout-review stitch data/real --workers 4                 # 提前把所有审核视频拼好
./rollout-review list data/real --verdict problem             # 输出形如 3,10-12
./rollout-review export data/real > judgements.csv
```

审核完之后把判为有问题的集去掉，比如配合 [lerobot-surgery](https://github.com/heliang-pu/lerobot-surgery)：

```bash
lerobot-surgery drop data/real data/real_clean \
    --drop "$(./rollout-review list data/real --verdict problem)" --reason "人工审核判为有问题"
```

## 键盘

| 键 | 作用 |
|---|---|
| 1 / 2 / 3 | 没问题 / 有问题 / 不确定（默认判完自动跳到下一集） |
| 0 | 清除结论 |
| J / K | 下一集 / 上一集 |
| 空格 | 播放或暂停 |
| ← / → | 快退或快进 1 秒 |

## 文件放在哪

都在数据集**旁边**的 `<数据集名>_review/` 目录里，不在数据集里面。原因是 lerobot 会把数据集目录下的每个 mp4 都算进去，
多一个文件就会出错。

| 文件 | 内容 |
|---|---|
| `episode_000012.mp4` | 第 12 集的审核视频（H.264，浏览器能直接播）。第一次打开时生成，源视频更新后自动重做 |
| `judgements.json` | 当前的全部结论：集号 → 结论、标签、备注、数据指纹、时间 |
| `judgements.log.jsonl` | 每一次修改的流水，只追加，用来追溯 |

## 结论什么时候会失效

结论绑定的是集号**和**这一集数据文件的指纹。数据集被删集重新编号、或者某一集被重录之后，同一个集号下已经是别的数据，
原来的结论不能再算数。这时页面上显示"已失效"，统计、`list` 和导出都不把它当作有效结论。

所以建议的顺序是：先审核，再根据结论生成一个**新的**数据集，不要在审核过的数据集上原地修改。

## 注意

- 页面没有登录。默认只监听 127.0.0.1；用 `--host 0.0.0.0` 时，同一网络里的人都能看和改结论。
- 相机不超过三个时横向排成一行，四个以上排成方阵；每个画面缩放到 480×360。

## 测试

```bash
python3 -m pytest -q
```

测试用 ffmpeg 生成的小视频，起一个真的 HTTP 服务，覆盖结论接口、结论失效和视频的分段读取（浏览器拖动进度条需要它）。
