# EndLoss Sequential Cross-Layer Quantization
## Global NLL end-loss → GuideQuant-style curvature → cross-layer error propagation → CD + exact codebook update

> **Mục tiêu**
>
> Tài liệu này mô tả một phiên bản implementation rõ ràng, tối giản và dễ kiểm tra của phương pháp:
>
> 1. xuất phát từ **NLL end-loss của toàn mô hình**;
> 2. bỏ raw calibration mean-gradient \(g\) khỏi objective;
> 3. dùng **GuideQuant-style empirical-Fisher curvature** để tạo \(H_{ll}\) cho từng linear layer / output group;
> 4. giữ thêm **signed output gradient** \(D_l\) để truyền quantization error từ các layer trước sang layer hiện tại;
> 5. quantize tuần tự từng layer bằng objective mới:
>
> \[
> \boxed{
> J_l(e_l)
> =
> \frac12 e_l^\top H_{ll} e_l
> +
> r_l^\top e_l
> }
> \]
>
> trong đó:
>
> \[
> \boxed{
> r_l \approx F_{l,<l}e_{<l}.
> }
> \]
>
> 6. solver dùng **coordinate descent (CD)** cho assignment và **closed-form linear solve** cho codebook;
> 7. initialization giữ **y hệt SqueezeLLM initialization đã có trong GuideQuant/LNQ**.

---

# 1. Source tham chiếu

GuideQuant:

- Paper: `https://arxiv.org/abs/2505.07004`
- Code: `https://github.com/snu-mllab/GuidedQuant`

Phần curvature trong tài liệu này giữ đúng tinh thần GuideQuant:

- dùng end-loss gradient;
- dùng block-diagonal empirical-Fisher approximation;
- dùng weighted activation covariance;
- output channels có thể được group để share một curvature matrix.

**Không dùng exact full Hessian của toàn model.**

---

# 2. Convention ma trận dùng trong tài liệu

Để tránh nhầm transpose, tài liệu này cố định convention của một `nn.Linear`:

\[
W_l\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}
\]

Calibration activations:

\[
X_l\in\mathbb R^{T\times d_{\text{in}}}
\]

Linear output:

\[
\boxed{
Z_l=X_lW_l^\top
}
\]

nên:

\[
Z_l\in\mathbb R^{T\times d_{\text{out}}}.
\]

Signed output gradient của end NLL:

\[
\boxed{
D_l=
\frac{\partial L_{\mathrm{NLL}}}{\partial Z_l}
}
\]

với:

\[
D_l\in\mathbb R^{T\times d_{\text{out}}}.
\]

Ở đây \(T\) là tổng số valid calibration tokens sau masking.

> **Lưu ý về \(XX^\top\) / \(X^\top X\)**
>
> Với convention \(X:[T,d_{\text{in}}]\), curvature có dạng:
>
> \[
> X^\top \operatorname{Diag}(s)X.
> \]
>
> Nếu code hiện tại lưu activation dưới dạng \([d_{\text{in}},T]\), cùng một phép tính sẽ xuất hiện dưới dạng:
>
> \[
> X\operatorname{Diag}(s)X^\top.
> \]
>
> Đây chỉ là khác convention lưu matrix, không phải khác phương pháp.

---

# 3. Xuất phát từ NLL end-loss

Gọi toàn bộ full-precision weights của model là:

\[
W.
\]

Sau quantization:

\[
\hat W=W+e
\]

với:

\[
\boxed{
e=\hat W-W.
}
\]

Ta xét NLL degradation:

\[
\boxed{
\Delta L
=
L_{\mathrm{NLL}}(\hat W)
-
L_{\mathrm{NLL}}(W).
}
\]

Taylor bậc hai quanh full-precision model:

\[
\Delta L
\approx
g^\top e
+
\frac12e^\top He.
\]

Trong phương pháp này ta **không dùng raw first-order calibration gradient**.

Working assumption:

\[
\boxed{
g\approx0.
}
\]

Do đó:

\[
\boxed{
\Delta L
\approx
\frac12e^\top He.
}
\]

## 3.1. Vì sao bỏ raw \(g\)?

Raw calibration gradient:

\[
\hat g
=
\frac1T\sum_t g_t
\]

được estimate từ calibration set nhỏ.

Trong các thử nghiệm trước, linear term này làm weight/codebook shift rất mạnh và có thể dẫn tới:

- codeword explosion;
- weight explosion;
- PPL rất lớn;
- NaN.

Method mới **không dùng**:

\[
\hat g^\top e.
\]

Nhưng vẫn giữ **per-token signed gradients** \(g_t\) để xây Fisher và cross-layer correlation.

Đây là hai thứ khác nhau:

\[
\boxed{
\text{bỏ mean gradient }\hat g
\neq
\text{bỏ signed per-token gradients}.
}
\]

---

# 4. Chia global error theo layer

Gọi:

\[
e=
\begin{bmatrix}
e_1\\
e_2\\
\vdots\\
e_L
\end{bmatrix}.
\]

Global quadratic curvature:

\[
H=
\begin{bmatrix}
H_{11} & H_{12} & \cdots\\
H_{21} & H_{22} & \cdots\\
\vdots & & \ddots
\end{bmatrix}.
\]

Khai triển:

\[
\Delta L
\approx
\frac12
\sum_l e_l^\top H_{ll}e_l
+
\sum_{l<m}
e_l^\top H_{lm}e_m.
\]

Hai loại term:

### Trong cùng layer

\[
H_{ll}
\]

đo độ nhạy của EndLoss đối với quantization error của chính layer \(l\).

### Giữa các layer

\[
H_{lm},\quad l\neq m
\]

đo coupling giữa error của hai layer.

---

# 5. Quantize tuần tự từng layer

Ta quantize theo thứ tự model.

Khi bắt đầu layer \(l\):

- các layer trước \(1,\ldots,l-1\) đã có quantization error;
- layer hiện tại \(l\) chưa quyết định;
- các layer sau vẫn tạm coi error bằng 0.

Với các error trước đã cố định, objective liên quan đến \(e_l\) là:

\[
\boxed{
J_l(e_l)
=
\frac12e_l^\top H_{ll}e_l
+
e_l^\top
\left(
\sum_{m<l}H_{lm}e_m
\right).
}
\]

Định nghĩa:

\[
\boxed{
r_l
=
\sum_{m<l}H_{lm}e_m.
}
\]

Vậy:

\[
\boxed{
J_l(e_l)
=
\frac12e_l^\top H_{ll}e_l
+
r_l^\top e_l.
}
\]

## 5.1. Layer đầu tiên

Chưa có error trước:

\[
r_1=0.
\]

Nên:

\[
\boxed{
J_1(e_1)
=
\frac12e_1^\top H_{11}e_1.
}
\]

## 5.2. Layer thứ hai

Sau khi layer 1 đã quantize:

\[
e_1\neq0.
\]

Ta có:

\[
\boxed{
r_2=H_{21}e_1.
}
\]

Layer 2 solve:

\[
\boxed{
J_2(e_2)
=
\frac12e_2^\top H_{22}e_2
+
r_2^\top e_2.
}
\]

## 5.3. Tổng quát

\[
\boxed{
r_l
=
H_{l1}e_1+\cdots+H_{l,l-1}e_{l-1}.
}
\]

Điểm quan trọng:

> **Ta không materialize các cross-layer matrices \(H_{lm}\).**
>
> Ta chỉ cần trực tiếp tính matrix-vector product:
>
> \[
> H_{l,<l}e_{<l}.
> \]

---

# 6. Dùng empirical Fisher thay cho global Hessian

Exact global Hessian quá lớn.

Ta dùng empirical Fisher approximation:

\[
\boxed{
H\approx F
=
\frac1T
\sum_{t=1}^{T}
g_tg_t^\top.
}
\]

Trong đó:

\[
g_t
=
\nabla_W \ell_t
\]

là **signed per-token gradient** của end NLL với toàn bộ model.

Chia theo layer:

\[
g_t=
\begin{bmatrix}
g_{t,1}\\
g_{t,2}\\
\vdots
\end{bmatrix}.
\]

Khi đó:

\[
\boxed{
F_{ll}
=
\frac1T\sum_t
g_{t,l}g_{t,l}^\top
}
\]

và:

\[
\boxed{
F_{lm}
=
\frac1T\sum_t
g_{t,l}g_{t,m}^\top.
}
\]

Ta dùng:

\[
\boxed{
H_{ll}\approx F_{ll},
\qquad
H_{lm}\approx F_{lm}.
}
\]

---

# 7. GuideQuant-style \(H_{ll}\)

Với linear layer:

\[
Z_l=X_lW_l^\top.
\]

Với output row/channel \(j\):

\[
w_{l,j}\in\mathbb R^{d_{\text{in}}}.
\]

Với token \(t\):

\[
d_{t,j}^{(l)}
=
\frac{\partial\ell_t}
{\partial z_{t,j}^{(l)}}.
\]

Gradient theo row \(w_{l,j}\):

\[
\boxed{
g_{t,l,j}
=
d_{t,j}^{(l)}x_t^{(l)}.
}
\]

Do đó empirical Fisher block của row \(j\):

\[
\begin{aligned}
H_{l,j}
&\approx
\frac1T
\sum_t
g_{t,l,j}g_{t,l,j}^\top
\\
&=
\frac1T
\sum_t
\left(d_{t,j}^{(l)}\right)^2
x_t^{(l)}x_t^{(l)\top}.
\end{aligned}
\]

Matrix form:

\[
\boxed{
H_{l,j}
=
\frac1T
X_l^\top
\operatorname{Diag}
\left(
D_l[:,j]^2
\right)
X_l.
}
\]

Đây là weighted \(X^\top X\) kiểu GuideQuant.

---

# 8. Output-channel grouping giống GuideQuant

GuideQuant không nhất thiết giữ một \(H\) riêng cho từng output row.

Cho output group \(J_k^{(l)}\):

\[
s_{l,k}[t]
=
\frac1{|J_k^{(l)}|}
\sum_{j\in J_k^{(l)}}
D_l[t,j]^2.
\]

Sau đó:

\[
\boxed{
H_{l,k}
=
\frac1T
X_l^\top
\operatorname{Diag}(s_{l,k})
X_l.
}
\]

Các row thuộc cùng group dùng chung:

\[
H_{l,k}.
\]

**Giữ nguyên grouping mechanism hiện tại của GuideQuant.**

Không cần thiết kế curvature mới ở version đầu.

---

# 9. Phần mới: phải giữ signed \(D_l\)

GuideQuant curvature chỉ cần:

\[
D_l^2.
\]

Nhưng khi square:

\[
(+d)^2=(-d)^2.
\]

Thông tin dấu bị mất.

Cross-layer propagation lại cần correlation kiểu:

\[
g_{t,l}g_{t,m}^\top,
\]

nên **bắt buộc cần signed gradient**.

Do đó calibration collection phải giữ thêm:

\[
\boxed{
D_l=
\frac{\partial L_{\mathrm{NLL}}}{\partial Z_l}
}
\]

trước khi square.

---

# 10. Không cần build \(F_{lm}\)

Ta cần:

\[
r_l
\approx
F_{l,<l}e_{<l}.
\]

Xét trước chỉ một previous layer \(m\):

\[
F_{lm}e_m
=
\frac1T
\sum_t
g_{t,l}
\left(
g_{t,m}^\top e_m
\right).
\]

Đặt scalar:

\[
\boxed{
c_t^{(m)}
=
g_{t,m}^\top e_m.
}
\]

Khi đó:

\[
\boxed{
F_{lm}e_m
=
\frac1T
\sum_t
c_t^{(m)}g_{t,l}.
}
\]

Với nhiều previous layers:

\[
\boxed{
c_t
=
\sum_{m<l}
g_{t,m}^\top e_m.
}
\]

Vậy:

\[
\boxed{
r_l
=
\frac1T
\sum_t
c_tg_{t,l}.
}
\]

Điểm quan trọng:

> Sau khi chấp nhận \(H\approx F\), phép tính này **không phải approximation mới**.
>
> Nó chỉ là cách tính:
>
> \[
> F_{l,<l}e_{<l}
> \]
>
> mà không materialize \(F_{lm}\).

---

# 11. Cách update accumulator \(c\) sau mỗi linear layer

Sau khi quantize layer \(l\):

\[
\hat W_l.
\]

Weight error:

\[
\boxed{
E_l=\hat W_l-W_l.
}
\]

Output error do chính linear layer đó:

\[
\boxed{
\Delta Z_l=X_lE_l^\top.
}
\]

Shape:

\[
\Delta Z_l\in\mathbb R^{T\times d_{\text{out}}}.
\]

Với mỗi token:

\[
g_{t,l}^\top e_l
=
D_l[t,:]^\top
\Delta Z_l[t,:].
\]

Do đó update accumulator:

\[
\boxed{
c_t
\leftarrow
c_t
+
D_l[t,:]^\top
\Delta Z_l[t,:].
}
\]

Vectorized:

\[
\boxed{
c
\leftarrow
c
+
\operatorname{rowsum}
\left(
D_l\odot \Delta Z_l
\right).
}
\]

Pseudocode:

```python
E = W_q - W                      # [d_out, d_in]
delta_Z = X @ E.T                # [T, d_out]

c += (D * delta_Z).sum(dim=1)    # [T]
```

Chỉ cần một scalar \(c_t\) cho mỗi calibration token.

---

# 12. Cách tính \(r_l\) trước khi quantize layer hiện tại

Trước layer \(l\), accumulator \(c\) đã chứa contribution từ tất cả previous layers.

Với row \(j\):

\[
g_{t,l,j}
=
D_l[t,j]x_t.
\]

Do đó:

\[
\boxed{
r_{l,j}
=
\frac1T
\sum_t
c_tD_l[t,j]x_t.
}
\]

Matrix form:

\[
\boxed{
r_{l,j}
=
\frac1T
X_l^\top
\left(
c\odot D_l[:,j]
\right).
}
\]

Tính tất cả output rows cùng lúc:

\[
\boxed{
R_l
=
\frac1T
\left(
D_l\odot c[:,None]
\right)^\top
X_l.
}
\]

Shape:

\[
R_l\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}.
\]

Row \(j\) của \(R_l\) chính là:

\[
r_{l,j}^\top.
\]

Pseudocode:

```python
R = (D * c[:, None]).T @ X
R /= num_valid_tokens
```

Đây chỉ là một GEMM lớn, rất phù hợp GPU.

---

# 13. Objective thực tế theo row

Theory dùng \(e_l\) của toàn layer.

GuideQuant-style approximation cho phép solve từng output row.

Với row \(j\):

\[
w=w_{l,j}
\]

quantized row:

\[
q=\hat w_{l,j}
\]

error:

\[
e=q-w.
\]

Lấy \(H\) là group Hessian của row đó:

\[
H=H_{l,k(j)}.
\]

Lấy:

\[
r=r_{l,j}.
\]

Objective:

\[
\boxed{
J(q)
=
\frac12(q-w)^\top H(q-w)
+
r^\top(q-w).
}
\]

Đây là objective solver phải tối ưu.

---

# 14. Non-uniform scalar quantization

Codebook:

\[
c=
[c_1,\ldots,c_K]^\top.
\]

Mỗi weight coordinate phải chọn đúng một codeword:

\[
q_i\in\{c_1,\ldots,c_K\}.
\]

Ta dùng alternating optimization:

1. giữ codebook cố định → update assignments bằng CD;
2. giữ assignments cố định → update codebook bằng exact small linear solve;
3. lặp lại theo stopping logic hiện tại của LNQ/GuideQuant.

---

# 15. Initialization

**Giữ nguyên SqueezeLLM initialization đã có trong GuideQuant/LNQ.**

Không thiết kế initialization mới trong version đầu.

Luồng:

```text
existing SqueezeLLM initialization
        ↓
initial assignments
        ↓
initial codebook
        ↓
new CD / codebook solver with propagated r
```

Layer đầu tiên:

\[
r_1=0.
\]

Do đó layer đầu gần như chính là curvature-only optimization.

---

# 16. CD assignment update

Với một row, define:

\[
e=q-w.
\]

Objective:

\[
J(q)
=
\frac12e^\top He+r^\top e.
\]

Maintain residual/gradient:

\[
\boxed{
s=He+r.
}
\]

Xét coordinate \(i\).

Current value:

\[
q_i.
\]

Candidate codeword:

\[
c_k.
\]

Đặt:

\[
\delta_k=c_k-q_i.
\]

Nếu chỉ đổi coordinate \(i\), exact objective change là:

\[
\boxed{
\Delta J_k
=
s_i\delta_k
+
\frac12H_{ii}\delta_k^2.
}
\]

Do đó assignment update:

\[
\boxed{
q_i^{\text{new}}
=
\arg\min_{c_k}
\left[
s_i(c_k-q_i)
+
\frac12H_{ii}(c_k-q_i)^2
\right].
}
\]

## 16.1. Recommended implementation

**Nên code trực tiếp \(\Delta J_k\)** cho tất cả \(K\) codewords.

Không cần divide bởi \(H_{ii}\).

Pseudocode:

```python
delta = codebook - q_i
cost = s_i * delta + 0.5 * H_ii * delta.square()

k_new = cost.argmin()
q_new = codebook[k_new]
```

Vì \(K=2^b\) nhỏ, ví dụ:

- 2-bit: \(K=4\)
- 3-bit: \(K=8\)
- 4-bit: \(K=16\)

nên evaluate toàn bộ codebook rất rẻ.

---

# 17. Equivalent nearest-codeword interpretation

Nếu:

\[
H_{ii}>0,
\]

coordinate optimum liên tục là:

\[
\boxed{
t_i
=
q_i-\frac{s_i}{H_{ii}}.
}
\]

Sau đó:

\[
q_i^{\text{new}}
=
\operatorname{nearest\_codeword}(t_i).
\]

Nhưng đây chủ yếu là **interpretation**.

### Không khuyến nghị dùng dạng chia trong code v1

Vì nếu:

\[
H_{ii}
\]

rất nhỏ, quantity:

\[
s_i/H_{ii}
\]

có thể rất lớn.

Dùng direct candidate cost:

\[
\Delta J_k
=
s_i\delta_k
+
\frac12H_{ii}\delta_k^2
\]

tránh phép chia này hoàn toàn.

Đây là một safeguard tự nhiên, không cần thêm hyperparameter.

---

# 18. Fast residual update trong CD

Sau khi coordinate \(i\) đổi:

\[
q_i^{old}\rightarrow q_i^{new},
\]

đặt:

\[
\delta
=
q_i^{new}-q_i^{old}.
\]

Ta có:

\[
s=H(q-w)+r.
\]

Không cần tính lại toàn bộ \(s\).

Chỉ update:

\[
\boxed{
s
\leftarrow
s
+
H_{:,i}\delta.
}
\]

Pseudocode:

```python
delta = q_new - q_old
q[i] = q_new
s += H[:, i] * delta
```

Một CD sweep đi qua:

\[
i=1,\ldots,d_{\text{in}}.
\]

Các output rows có thể chạy song song trên GPU.

---

# 19. Codebook update

Giữ assignments cố định.

Dùng assignment matrix:

\[
P\in\{0,1\}^{d_{\text{in}}\times K}
\]

mỗi row của \(P\) có đúng một số 1.

Quantized row:

\[
\boxed{
q=Pc.
}
\]

Error:

\[
e=Pc-w.
\]

Objective:

\[
J(c)
=
\frac12(Pc-w)^\top H(Pc-w)
+
r^\top(Pc-w).
\]

Đạo hàm:

\[
\nabla_cJ
=
P^\top H(Pc-w)
+
P^\top r.
\]

Cho bằng 0:

\[
P^\top HPc
-
P^\top Hw
+
P^\top r
=
0.
\]

Suy ra:

\[
\boxed{
(P^\top HP)c
=
P^\top(Hw-r).
}
\]

Đặt:

\[
\boxed{
A=P^\top HP
}
\]

và:

\[
\boxed{
b=P^\top(Hw-r).
}
\]

Solve:

\[
\boxed{
Ac=b.
}
\]

Vì:

\[
A\in\mathbb R^{K\times K},
\]

đây là một hệ rất nhỏ.

Ví dụ 3-bit:

\[
K=8
\]

chỉ cần solve hệ \(8\times8\).

---

# 20. Không materialize \(P\)

Trong code không cần tạo dense one-hot matrix \(P\).

Chỉ cần assignment index:

```text
assignment[i] = k
```

nghĩa là coordinate \(i\) đang dùng codeword \(k\).

Vế phải:

\[
b=P^\top(Hw-r)
\]

tương đương group-sum theo assignment.

Nếu:

\[
v=Hw-r,
\]

thì:

\[
\boxed{
b_k
=
\sum_{i:\,assignment[i]=k}
v_i.
}
\]

Dùng `scatter_add`.

Tương tự:

\[
A_{ab}
=
\sum_{i:\,assignment[i]=a}
\sum_{j:\,assignment[j]=b}
H_{ij}.
\]

Có thể dùng indexing/scatter hoặc logic hiện tại của LNQ.

---

# 21. Không dùng explicit matrix inverse

Công thức toán có thể viết:

\[
c=A^\dagger b.
\]

Nhưng code không nên tạo:

```python
torch.inverse(A)
```

Nên dùng linear solve:

```python
torch.linalg.solve(...)
```

khi nonsingular, hoặc existing robust solve / `lstsq` logic của LNQ khi singular/rank-deficient.

Không thêm balancing coefficient mới.

---

# 22. Sau codebook update

Codebook update làm nhiều coordinates của \(q\) thay đổi cùng lúc.

Do đó sau khi có codebook mới:

\[
q=Pc,
\]

nên recompute:

\[
\boxed{
s=H(q-w)+r
}
\]

từ đầu.

Lý do:

- tránh numerical drift do nhiều incremental updates;
- cost chỉ là một matrix-vector product;
- làm CD sweep tiếp theo sạch và reproducible hơn.

---

# 23. Full solver cho một row

```text
Input:
    w
    H
    r
    initial assignments from SqueezeLLM
    initial codebook from SqueezeLLM

Build:
    q from assignments + codebook
    s = H(q-w) + r

Repeat using existing LNQ stopping logic:

    A. CD assignment sweep
        for i = 1 ... d_in:
            evaluate all K candidate codewords:
                delta_k = c_k - q_i

                ΔJ_k =
                    s_i * delta_k
                    + 0.5 * H_ii * delta_k^2

            choose argmin
            update q_i
            update s incrementally:
                s += H[:, i] * delta

    B. Exact codebook update
        build A = P^T H P
        build b = P^T(Hw-r)
        solve A c = b

        rebuild q = P c
        recompute:
            s = H(q-w) + r

Output:
    q
```

---

# 24. Full model algorithm

## Phase A — collect statistics một lần trên FP model

Chạy calibration trên full-precision model.

Với mỗi quantized linear module \(l\):

1. cache / obtain:
   \[
   X_l
   \]
2. backward end NLL;
3. cache signed:
   \[
   D_l
   \]
4. build GuideQuant squared-gradient statistics:
   \[
   D_l^2
   \]
5. build group curvature:
   \[
   H_{l,k}
   =
   \frac1T
   X_l^\top
   \operatorname{Diag}(s_{l,k})
   X_l.
   \]

Tất cả \(H_l\) có thể tính trước.

**Không cần recompute Hessian sau mỗi quantized layer.**

Lý do: quadratic approximation đang được xây quanh cùng full-precision model \(W\).

---

## Phase B — sequential quantization

Khởi tạo:

\[
\boxed{
c=0.
}
\]

Sau đó đi layer/module theo đúng quantization order.

### Trước layer \(l\)

Tính:

\[
\boxed{
R_l
=
\frac1T
(D_l\odot c[:,None])^\top X_l.
}
\]

Mỗi row \(j\):

\[
r_{l,j}=R_l[j,:].
\]

### Quantize layer \(l\)

Cho mỗi output row \(j\):

- lấy group Hessian tương ứng \(H_{l,k(j)}\);
- lấy \(r_{l,j}\);
- initialize bằng SqueezeLLM;
- solve bằng CD + exact codebook update.

Sau khi tất cả rows của linear layer hoàn tất:

\[
E_l=\hat W_l-W_l.
\]

Tính:

\[
\Delta Z_l=X_lE_l^\top.
\]

Update:

\[
\boxed{
c
\leftarrow
c+
\operatorname{rowsum}
(D_l\odot\Delta Z_l).
}
\]

Sau đó sang layer tiếp theo.

---

# 25. Toàn pipeline

```text
                    FULL-PRECISION MODEL
                            │
                 calibration + end NLL
                            │
                         backward
                            │
              ┌─────────────┴─────────────┐
              │                           │
             X_l                     signed D_l
              │                           │
              │                       square D_l
              │                           │
              └──────────────┬────────────┘
                             │
                  GuideQuant curvature
                             │
                  H_1, H_2, ..., H_L
                             │
                   all computed once
                             │
──────────────────────── sequential quantization ────────────────────────
                             │
                           c = 0
                             │
                       current layer l
                             │
                R_l = (D_l * c)^T X_l / T
                             │
                       row-wise r_l
                             │
             H_l + r_l + SqueezeLLM initialization
                             │
               CD assignment + codebook update
                             │
                         W_hat_l
                             │
                    E_l = W_hat_l - W_l
                             │
                    ΔZ_l = X_l E_l^T
                             │
             c += rowsum(D_l * ΔZ_l)
                             │
                         next layer
```

---

# 26. Fast implementation notes

## 26.1. Hessian collection vẫn tính trước như GuideQuant

Không cần:

```text
compute H1
quantize layer 1
compute H2
quantize layer 2
...
```

Có thể giữ:

```text
collect all GuideQuant H first
then sequential quantization
```

Thứ thay đổi sau mỗi layer là:

\[
c
\]

và:

\[
r_l,
\]

không phải \(H_l\).

---

## 26.2. Cross-layer propagation không build cross Hessian

Không bao giờ materialize:

\[
F_{21},F_{31},F_{32},\ldots
\]

Chỉ giữ:

\[
c\in\mathbb R^T.
\]

Sau đó current-layer propagation:

\[
R_l
=
(D_l\odot c[:,None])^\top X_l/T.
\]

Đây là một GEMM.

---

## 26.3. Signed \(D_l\) là bắt buộc

Chỉ lưu:

\[
D_l^2
\]

là không đủ.

Phải giữ signed:

\[
D_l.
\]

Nếu memory GPU không đủ:

- giữ \(H_l\) như hiện tại;
- offload \(X_l,D_l\) sang CPU/disk;
- khi quantize layer \(l\), load đúng \(X_l,D_l\) của layer đó;
- không cần giữ tất cả \(X_l,D_l\) cùng lúc trên GPU.

Không thay đổi toán học.

---

## 26.4. Normalization phải thống nhất

Cả:

\[
H_l
\]

và:

\[
r_l
\]

phải dùng cùng:

- valid-token mask;
- calibration examples;
- token count \(T\);
- gradient scaling convention;
- dtype accumulation convention.

Nếu GuideQuant có internal scaling trước khi square gradient, cần đảm bảo signed \(D_l\) dùng cho propagation được đưa về scale tương thích.

Không được để:

\[
H_l
\]

và:

\[
r_l
\]

khác normalization.

---

# 27. Các safeguard quan trọng để tránh weight explosion

## 27.1. Không đưa raw calibration mean-gradient trở lại

Không thêm:

\[
\hat g^\top e.
\]

Objective chỉ là:

\[
\boxed{
\frac12e^\top He+r^\top e.
}
\]

Trong đó \(r\) chỉ xuất hiện sau khi previous layers thật sự có quantization error.

---

## 27.2. Không dùng shifted continuous target trong implementation

Có thể về đại số viết continuous optimum liên quan tới:

\[
-H^{-1}r,
\]

nhưng **không dùng nó làm target để quantize**.

Không code:

```python
target = w - solve(H, r)
```

rồi quantize target.

Version này dùng trực tiếp original objective bằng CD.

---

## 27.3. Assignment CD nên dùng direct candidate cost

Ưu tiên:

\[
\Delta J_k
=
s_i\delta_k
+
\frac12H_{ii}\delta_k^2
\]

thay vì:

\[
q_i-s_i/H_{ii}.
\]

Lý do:

- không divide bởi diagonal rất nhỏ;
- exact cho coordinate;
- \(K\) nhỏ nên rất rẻ;
- giảm một failure mode có thể làm numerical target rất lớn.

---

## 27.4. Không normalize/clip \(r\) bằng hyperparameter mới

Version đầu không thêm:

- \(\alpha r\);
- manual clipping;
- manually tuned damping riêng cho propagated error.

Nếu \(r\) quá lớn, trước tiên debug:

- normalization mismatch;
- signed gradient scaling;
- token masking;
- \(X/D\) alignment;
- error accumulation;
- Hessian group mapping.

---

## 27.5. Reuse existing GuideQuant/LNQ numerical safeguards

Giữ nguyên:

- Hessian damping hiện tại;
- Cholesky / solver safeguards hiện tại;
- empty-cluster handling hiện tại;
- stopping logic hiện tại.

Không tạo thêm một tunable numerical hyperparameter mới ở version đầu.

---

## 27.6. Recompute residual sau codebook update

Sau mỗi codebook solve:

\[
s=H(q-w)+r
\]

nên recompute từ đầu để tránh accumulated floating-point drift.

---

## 27.7. Finite-value assertions

Trong debug mode, assert:

```text
isfinite(H)
isfinite(r)
isfinite(codebook)
isfinite(q)
isfinite(c)
```

Nếu xuất hiện NaN/Inf, dừng tại layer đầu tiên gây lỗi.

Không để NaN propagate tới PPL evaluation mới tìm lỗi.

---

# 28. Debug statistics nên log

Cho mỗi layer/group:

### Curvature

```text
trace(H)
||H||_F
min(diag(H))
median(diag(H))
max(diag(H))
```

### Propagation

```text
||r||_2
max_abs(r)
mean_abs(r)
```

### Error accumulator

```text
mean(c)
std(c)
max_abs(c)
```

### Weight/codebook

```text
min/max original weight
min/max quantized weight
min/max codebook
max_abs(W_hat - W)
```

### Objective

Trước và sau mỗi CD sweep / codebook update:

\[
J(q)
=
\frac12(q-w)^\top H(q-w)+r^\top(q-w).
\]

Objective không được tăng sau exact coordinate updates và exact codebook solve, ngoài floating-point noise rất nhỏ.

---

# 29. Unit tests bắt buộc

## 29.1. Layer 1 regression

Vì:

\[
r_1=0,
\]

solver layer 1 phải reduce về curvature-only LNQ-like objective:

\[
\frac12e^\top He.
\]

---

## 29.2. Zero previous error

Nếu:

\[
E_1=\cdots=E_{l-1}=0,
\]

thì:

\[
c=0
\]

và:

\[
r_l=0.
\]

---

## 29.3. Matrix-free cross-Fisher test trên toy model

Trên model rất nhỏ, explicit build:

\[
F_{lm}
=
\frac1T
\sum_t
g_{t,l}g_{t,m}^\top.
\]

So sánh:

\[
F_{lm}e_m
\]

với accumulator implementation:

\[
\frac1T
\sum_t
g_{t,l}
(g_{t,m}^\top e_m).
\]

Hai kết quả phải match numerical precision.

Đây là test cực quan trọng.

---

## 29.4. Accumulator identity test

Với linear layer:

\[
g_{t,l}^\top e_l
\]

phải bằng:

\[
D_l[t,:]^\top
(X_lE_l^\top)[t,:].
\]

Test trực tiếp trên tensor nhỏ.

---

## 29.5. CD coordinate exactness

Với một coordinate \(i\), brute-force evaluate full objective cho tất cả \(K\) codewords.

Kết quả argmin phải giống:

\[
\arg\min_k
\left[
s_i\delta_k+\frac12H_{ii}\delta_k^2
\right].
\]

---

## 29.6. Codebook closed-form test

Fix assignments \(P\).

Sau khi solve:

\[
(P^\top HP)c=P^\top(Hw-r),
\]

gradient theo \(c\):

\[
P^\top H(Pc-w)+P^\top r
\]

phải gần zero.

---

# 30. Recommended implementation order

## Step 1 — Giữ GuideQuant curvature untouched

Trước tiên verify code hiện tại vẫn tạo đúng:

\[
H_{l,k}.
\]

Không thay solver hoặc propagation ngay.

---

## Step 2 — Bổ sung signed \(D_l\)

Hook cùng vị trí GuideQuant đang collect output gradient.

Giữ thêm signed tensor:

\[
D_l.
\]

Verify:

```text
D_l^2
```

tạo đúng squared-gradient stats cũ.

---

## Step 3 — Implement accumulator \(c\)

Ban đầu:

```python
c = zeros(num_valid_tokens)
```

Sau một quantized layer:

```python
E = W_q - W
delta_Z = X @ E.T
c += (D * delta_Z).sum(dim=1)
```

Toy-test identity trước khi chạy LLM.

---

## Step 4 — Implement \(R_l\)

```python
R = (D * c[:, None]).T @ X
R /= num_valid_tokens
```

Verify với explicit toy cross-Fisher.

---

## Step 5 — Modify solver objective

Từ curvature-only:

\[
\frac12e^\top He
\]

thành:

\[
\boxed{
\frac12e^\top He+r^\top e.
}
\]

---

## Step 6 — Implement CD assignment

Maintain:

```python
s = H @ (q - w) + r
```

Coordinate candidate cost:

```python
delta = codebook - q_i
cost = s_i * delta + 0.5 * H_ii * delta.square()
```

---

## Step 7 — Implement codebook update

Build:

\[
A=P^\top HP
\]

\[
b=P^\top(Hw-r)
\]

solve:

\[
Ac=b.
\]

Reuse LNQ/SqueezeLLM assignment structure.

---

## Step 8 — Integrate sequential loop

```text
c = 0

for layer in quantization_order:
    load H_l
    load X_l
    load signed D_l

    compute R_l from c

    quantize all rows using:
        H_group(row)
        r_row
        SqueezeLLM init
        CD + exact codebook update

    E_l = W_hat_l - W_l
    update c

    continue
```

---

# 31. Những gì KHÔNG làm ở version đầu

Không thêm:

- raw \(g_{\mathrm{NLL}}\);
- KL dual curvature;
- \(H_{\mathrm{NLL}}+H_{\mathrm{KL}}\);
- K-FAC;
- D+UUᵀ;
- new MM solver;
- new initialization;
- propagated-error scaling coefficient;
- manual clipping coefficient;
- recompute Hessian tại partial-quantized model.

Mục tiêu version đầu là isolate đúng contribution:

\[
\boxed{
\text{cross-layer EndLoss error propagation}
}
\]

trên GuideQuant-style curvature.

---

# 32. Công thức cần nhớ

## EndLoss quadratic

\[
\boxed{
L_{\mathrm{NLL}}(\hat W)-L_{\mathrm{NLL}}(W)
\approx
\frac12e^\top He
}
\]

## Current-layer objective

\[
\boxed{
J_l(e_l)
=
\frac12e_l^\top H_{ll}e_l+r_l^\top e_l
}
\]

## Propagated term

\[
\boxed{
r_l
\approx
F_{l,<l}e_{<l}
}
\]

## GuideQuant row/group curvature

\[
\boxed{
H_{l,k}
=
\frac1T
X_l^\top
\operatorname{Diag}(s_{l,k})
X_l
}
\]

## Signed-gradient accumulator

\[
\boxed{
c
\leftarrow
c+
\operatorname{rowsum}
\left[
D_l\odot(X_lE_l^\top)
\right]
}
\]

## Current-layer propagated matrix

\[
\boxed{
R_l
=
\frac1T
(D_l\odot c[:,None])^\top X_l
}
\]

## CD assignment candidate cost

\[
\boxed{
\Delta J_k
=
s_i(c_k-q_i)
+
\frac12H_{ii}(c_k-q_i)^2
}
\]

với:

\[
\boxed{
s=H(q-w)+r.
}
\]

## Codebook update

\[
\boxed{
(P^\top HP)c
=
P^\top(Hw-r).
}
\]

---

# 33. One-paragraph instruction cho Codex

> Keep the existing GuideQuant curvature collection and SqueezeLLM/LNQ initialization. During the full-precision NLL calibration backward, additionally preserve the signed output gradients \(D_l=\partial L/\partial Z_l\) before squaring. Build each layer/group curvature exactly as GuideQuant does. Quantization must then run sequentially. Maintain a per-token scalar accumulator \(c\), initially zero. Before quantizing layer \(l\), compute the row-wise propagated linear term \(R_l=(D_l\odot c[:,None])^\top X_l/T\). For each output row, solve \(J(q)=\frac12(q-w)^\top H(q-w)+r^\top(q-w)\) using cyclic coordinate descent with exact candidate cost \(\Delta J=s_i\delta+\frac12H_{ii}\delta^2\), where \(s=H(q-w)+r\), alternating with the exact codebook solve \((P^\top HP)c=P^\top(Hw-r)\). After the full linear layer is quantized, compute \(E_l=\hat W_l-W_l\), \(\Delta Z_l=X_lE_l^\top\), and update the accumulator \(c\leftarrow c+\operatorname{rowsum}(D_l\odot\Delta Z_l)\). Do not reintroduce the raw calibration mean-gradient, do not form a shifted target using \(H^{-1}r\), and do not add new balancing/clipping hyperparameters in the first implementation.
