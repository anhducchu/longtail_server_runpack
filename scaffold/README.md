# Frozen-score scaffold

Đây là experiment stage-2 độc lập. Nó chỉ đọc checkpoint baseline và dữ liệu;
mọi file được tạo đều nằm trong `scaffold/outputs/<name>/`. Nó không sửa
`main.py`, `experiment.py`, config baseline hay source trong `third_party/`.

## Giả thuyết

Giữ nguyên score `s_theta` từ `ema_model` và chỉ học một MLP nhỏ
`rho_phi(y,t)` có thể phân bổ lại stochasticity cho long-tail data mà không
đánh đổi bằng việc train thêm score network.

So sánh bắt buộc dùng cùng checkpoint, label, seed và reverse integrator:

- baseline: `rho=1`;
- scaffold: `rho=rho_phi(y,t)`.

`rho_phi` khởi tạo chính xác tại 1. Optimizer chỉ nhận `phi`; mỗi update đều
kiểm tra không parameter nào của score nhận gradient.

## Chạy

Đặt checkpoint đã train ở bất kỳ đâu; scaffold chỉ đọc file đó. Từ repo root:

```bash
source .venv/bin/activate

python -m scaffold.run plan \
  --checkpoint runs/.../ddpm/seed_0/ckpt_300000.pt \
  --name ddpm_seed0

CUDA_VISIBLE_DEVICES=0 python -m scaffold.run all \
  --checkpoint runs/.../ddpm/seed_0/ckpt_300000.pt \
  --name ddpm_seed0
```

Đọc code theo thứ tự:

```text
run.py       chọn stage, hiện rõ train -> sample -> evaluate
train.py     load data và chỉ optimize phi
sample.py    sinh cặp rho=1 / rho=rho_phi rồi evaluate
core.py      ScalarScaffold và hai tích phân forward/reverse
common.py    load config/checkpoint, I/O an toàn dưới outputs/
```

Có thể chạy từng stage:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold.run train \
  --checkpoint runs/.../ckpt_300000.pt --name ddpm_seed0

CUDA_VISIBLE_DEVICES=0 python -m scaffold.run sample \
  --checkpoint runs/.../ckpt_300000.pt --name ddpm_seed0
```

Train tự resume từ `phi.pt`. Sample tạo hai condition bằng paired random noise.

## Output

```text
scaffold/outputs/ddpm_seed0/
├── config.resolved.json
├── phi.pt
├── training.csv
└── samples/
    ├── labels.npy
    ├── rho_1.npy
    ├── rho_1.provenance.json
    ├── learned.npy
    └── learned.provenance.json
```

`outputs/` được tạo tự động khi chạy và bị Git ignore toàn bộ.

## Evaluation chính thức

Config mặc định là screening: 200 update và 10.000 sample mỗi condition. Không
được dùng nó làm paper result. Để chạy evaluator canonical:

1. đổi `sampling.num_images` thành `50000`;
2. đổi `evaluation.enabled` thành `true`;
3. chạy lại `all`, hoặc chạy riêng `evaluate` sau khi đã có 50k sample.

```bash
python -m scaffold.run evaluate \
  --checkpoint runs/.../ckpt_300000.pt --name ddpm_seed0
```

Metric và feature cache vẫn được ghi dưới `scaffold/outputs/<name>/metrics/`.

## Fixed A/A0 sweep

Sweep `A/A0 = {0, 0.25, 0.5, 1, 2}` không train thêm parameter nào:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold.run fixed-sweep \
  --config scaffold/config.fixed_sweep.yaml \
  --checkpoint /absolute/path/ckpt_300000.pt \
  --name ddpm_300k_fixed_a_sweep
```

Năm condition dùng cùng labels và random-noise seed. Command có thể chạy lại:
sample/metric hoàn tất được giữ nguyên. Bảng kết quả được ghi thành JSON, CSV và
Markdown trong output. `Class-valid` được để `N/A` nếu chưa cung cấp một frozen
CIFAR-100 classifier độc lập; requested conditioning label không được coi là
bằng chứng ảnh đúng lớp.

Để chạy đồng thời năm condition trên cùng GPU, mở năm process `fixed-condition`
với cùng config, checkpoint và `--name`; mỗi process nhận một trong các tên
`pf_ode`, `reduced_025`, `reduced_05`, `standard_sde`, `increased_2`. Sau khi cả
năm kết thúc, chạy `fixed-evaluate` một lần để evaluate tuần tự và ghép bảng.

Command gọn hơn dưới đây tự xếp hàng tối đa hai condition cùng lúc (khoảng 10 GB
trên H100 hiện tại), sau đó evaluate và ghép bảng tự động:

```bash
python -m scaffold.run fixed-parallel \
  --parallel-workers 2 \
  --config scaffold/config.fixed_sweep.yaml \
  --checkpoint /absolute/path/ckpt_300000.pt \
  --name ddpm_300k_fixed_a_sweep
```

## Tiêu chí đọc kết quả

- Primary: Few-class FID của `learned` tốt hơn `rho_1` trên paired seeds.
- Secondary: FID/KID tổng và Many/Medium FID không suy giảm lớn.
- Anti-metric: training loss thấp không tự nó chứng minh generation tốt hơn.
- Tối thiểu 3 score seeds trước khi đưa ra kết luận.

## Kiểm thử

```bash
python -m pytest -q scaffold/test_scaffold.py
```

Test kiểm tra `rho=1` lúc khởi tạo, theta thật sự frozen, gradient vẫn tới phi,
paired sampling deterministic và output không thể thoát khỏi `scaffold/outputs`.
