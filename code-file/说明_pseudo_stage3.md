- 先生成伪标签：
```python
python birdclef2026_gm_make_pseudo_labels.py \
  --model-root outputs/birdclef2026_gm/你的teacher_run目录
```

- 如果你后面想用多个强模型一起当 teacher：
```python
python birdclef2026_gm_make_pseudo_labels.py \
  --model-root outputs/birdclef2026_gm/run_a \
  --model-root outputs/birdclef2026_gm/run_b
```

- 生成完以后，再做 stage3：
```python
python birdclef2026_gm_train_stage3_pseudo.py \
  --student-run-dir outputs/birdclef2026_gm/你的student_run目录 \
  --pseudo-root outputs/pseudo_labels/刚生成的pseudo目录
```

- 如果你想先稳一点，推荐第一轮可以这样：
```python
python birdclef2026_gm_train_stage3_pseudo.py \
  --student-run-dir outputs/birdclef2026_gm/你的student_run目录 \
  --pseudo-root outputs/pseudo_labels/刚生成的pseudo目录 \
  --stage3-epochs 6 \
  --stage3-backbone-lr 2e-5 \
  --stage3-head-lr 2e-4 \
  --pseudo-loss-weight 0.5
```