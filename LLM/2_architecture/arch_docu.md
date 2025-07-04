
# 개요

- Isotropic Architecture 사용이 대세
- 블록의 인풋과 아웃풋 텐서 shape가 같다 : [B, L, C]
- 디자인 스페이스를 보면 바뀔 수 있는 곳 위주로 정리

![Image](https://github.com/user-attachments/assets/8ede509a-9689-41eb-844a-2d99879068d4)


# Positional Embedding

트랜스포머에선 Self-Attention 연산으로 인해 단어의 순서 정보가 사라지는 Permutation invariance 성질이 있다. 그런데 언어는 시간의 순서에 따른 정보를 내재하고 있으므로 이에 대응되는 디자인이 필요함. 이를 위해 Position 관련 정보를 더해줘야함.  

다양한 기법들이 발전해왔음.


| 구분  | 기법                                                    | 주요 특징                      | 위치 정보 유형             |       적용 위치   |   인코딩 방식    |
| --- | ----------------------------------------------------- | -------------------------- | -------------------- | ------------ |  ------ | 
| 1세대 | **Sinusoidal (Vaswani et al., 2017)**                 | 위치에 따라 정적 사인파/코사인파 주입      | **절대 위치**            |   input + PE | 절대 위치, 비학습  |
| 2세대 | **Learned Positional Embedding**                      | 위치마다 학습 가능한 임베딩            | 절대 위치 (학습 기반)        | input + PE     | 절대 위치, 학습   | 
| 3세대 | **Relative Positional Encoding (Transformer-XL, T5)** | token 간 거리 기반 attention 수정 | **상대 위치**            |    logits  | 상대 위치, 학습  |
| 4세대 | **RoPE (Su et al., 2021)**                            | 복소수 회전 기반 상대 위치 내재화        | 상대 위치 (회전 기반)        |  Q, K   | 상대 위치, 비학습  |
| 5세대 | **ALiBi (Press et al., 2022)**                        | attention logits에 선형 벌점 추가 | 상대 위치 (no embedding) |  logits  | 상대 위치, 비학습  |
| 6세대 | **xPos (Sun et al., 2023)**                           | RoPE 확장, 길이 일반화 성능 강화      | 상대 위치 (회전 개선)        | Q, K  | RoPE 개선    |
| 7세대 | **RoPE v2 (Chen et al., 2024)**                       | 로터리 각도에 길이-비례 스케일 추가       | 상대 위치 (고성능 회전)       | Q, K   | RoPE + interpolation 강화 | 

---

### 1. Sinusoidal Positional Encoding 

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right),\quad
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

* **장점**: 사전 계산 가능, generalization 가능
* **단점**: 학습 불가능, 구조 유연성 부족


###  2. Learned Positional Embedding

* 위치마다 학습 가능한 벡터 $\text{PE}_p \in \mathbb{R}^d$

* GPT‑2 등 대부분의 decoder-only LLM에서 사용

* **장점**: 학습 데이터에 최적화

* **단점**: 시퀀스 길이 일반화 불가


### 3. Relative Positional Encoding (Transformer-XL, T5)

* **Transformer-XL**: $QK^T$ 계산에 상대 위치 벡터를 추가

* **T5**: Learnable relative bias $a_{i-j}$를 attention logit에 직접 더함

* **장점**: 긴 시퀀스에서 위치 차이에 민감한 표현 가능

* **단점**: 연산 복잡도 증가

### 4. RoPE

기본 원리는 벡터를 각도에 따라 회전시킨다는 것인데 이 때 각도는 시퀀스의 나중 토큰일 수록 회전각도를 더 크게 함. 벡터에 $e^{i\theta}$를 곱하면 회전시킬 수 있음. 특히 $e^{i\theta}$를 곱하면 벡터 크기 변화는 없이 회전 가능.

그런데 이런 복소수를 쓰는 회전은 2차원 상의 회전임. hidden state의 emb는 그런데 딱 2개만 있는게 아님. 2개씩 짝짓는 아이디어. 

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*6OZJXSTvdea1FQYag4I86Q.png)

이때 하나의 벡터 내에서 emb 차원 방향으로 바뀌는걸 고려하는 것에 더해서 시퀀스 길이 방향으로도 바뀌는 것을 고려해서 $\theta$가 달리지는 식으로 디자인함. 즉 `[B, H, T, Emb]`에서 `T`, `Emb`에 따라서 유니크한 $\theta$값이 부여되도록 함. `T` 방향으로 갈수록 각도가 커지고 `Emb` 방향에선 저차원일수록 각도가 크게함.

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*c3PUbEVx-77OidtZ3xKGfA.png)

이는 다음 공식에서 $k$와 $i$로 조절하게 됨.

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*zwA7Zq8grzX5Kue318obFA.png)

이를 좀 더 예쁘게 정리하면 다음과 같이 됨

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*LqgBXLOIbKFG_scC2fXbTw.png)

특정한 $k$ 번째 토큰에서 `Emb` 방향으로 갈수록 각도 변화를 보면 후반으로 갈수록 천천히 바뀜 (기울기)

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vflR_CK56j6-qHDpNxyYwg.png)

이때 시퀀스의 변화로 따져보면 시퀀스 후반에 위치할 수록 각도가 더 커지게 됨을 알 수 있음.

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EXgpitu4CINKQpo6L7m_Wg.png)

원본 Transformer에서 임베딩에 Positional Encoding을 더하여서 임베딩의 의미가 변화되었는데 RoPE에선 위치 관련 정보를 조절하는걸 Attention에 적용하여 임베딩은 건들지 않게 된다. (Value는 건들지 않으므로) 

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*4CY81e0potrNSer02qj-Pg.png)


기존 Attention 계산에선 위치의 상대정보가 고려되어 계산되지 않았지만 이게 고려되게 할 수 있음. 같은 단어여도 시퀀스 상의 위치에 따라 회전으로 벡터가 조절되게 되고 Dot product할때 이렇게 조절된 벡터간의 연산을 하게 됨. 같은 위치, 같은 단어면 당연하게도 유사도도 가장 높게 나옴. 

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*La2o6Wn_wVmxSKHrojfQlA.png
)

Base 값은 나중에 알게된것인데 10_000이 최적값이 아니라는게 밝혀짐. “Scaling Laws of RoPE-based Extrapolation” 논문에 따르면 10_000의 경우 Perplexity 값이 높아져버리는 현상. Llama-3 부턴 500_000로 셋팅해서 학습함. 

![Image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*sBa6aNBvJBpjC4-jqHCJ-g.png)

`Emb` 짝꿍 차원에서 낮은 짝일수록 주파수로 높고 더 위로 갈수록 주파수가 낮다. $\cos(2\pi f t) = \cos(\omega t)$ 에서 $f$가 높을수록 더 빠르게 값의 변화가 있음. llama3에선 이와 관련된 개선사항이 있음. 고주파 영역은 그대로 유지하도록 함. 반면에 저주파 영역은 파장길이를 늘려서 주기성을 없어도록 함. 자세한 내용은 여기를 참조 : https://medium.com/@hugmanskj/mastering-llama-rotary-positional-embedding-rope-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-9b1963a22852

예를 들면 k 번째 시퀀스 위치의 벡터 0, 1짝꿍은 다음과 같이 계산
$$
x_{k,0}^{new} = x_{k, 0}cosθ_{k, 0} −x_{k, 1}sinθ_{k, 0} \\
x_{k,1}^{new} = x_{k, 0}sinθ_{k, 0} +x_{k, 1}cosθ_{k, 0}
$$

일반화된 계산식은 다음처럼

$$
\begin{bmatrix}
x_{2i}^{\text{new}} \\
x_{2i+1}^{\text{new}}
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta_{k,i} & -\sin \theta_{k,i} \\
\sin \theta_{k,i} & \cos \theta_{k,i}
\end{bmatrix}
\cdot
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
$$

일반적인 각도 공식은 다음처럼

$$
\theta_{k,i} = \frac{k}{10000^{2i/d}} = k \cdot \omega_i
\quad\text{(with } \omega_i = 1 / 10000^{2i/d} \text{)}
$$


그런데 실제 많이 쓰는 구현 방식에선 `Emb` 방향에서 i, i+1 이랑 짝을 짓는게 아니라 절반을 쪼개서 90도 회전한 통벡터를 만들어서 계산한다. 어차피 Q,K 에서 같은 방식으로 짝이 지어기지만 하면 굳이 바로 옆과 짝을 지을 필요가 없기 때문. 바로 옆과 짝을 짓는 interleaved 방식 대신에 split-in-half 방식은 계산 효율적이라 많이 쓰게 됨. x[...,0::2] 처럼 stride가 2인 gather는 메모리 접근이 산발적이라 느림. chunk(2)로 앞/뒤 절반을 연속 메모리로 가져오면 캐시 효율이 높고 JIT Fusion(Flash-Attention 등)에도 유리

$$
R(\theta)=
\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{bmatrix}
$$

#### (2) 벡터 $[x_e,x_o]^\top$ 에 곱하면:

$$
\begin{aligned}
\begin{bmatrix}
x_e^{\text{new}}\\[2pt]
x_o^{\text{new}}
\end{bmatrix}
&=
R(\theta)
\begin{bmatrix}
x_e\\ x_o
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta\,x_e-\sin\theta\,x_o\\
\sin\theta\,x_e+\cos\theta\,x_o
\end{bmatrix}
\end{aligned}
$$


$$
=
\underbrace{
\begin{bmatrix}x_e\\x_o\end{bmatrix}}_{\text{원본}}
\cos\theta
\;+\;
\underbrace{
\begin{bmatrix}-x_o\\x_e\end{bmatrix}}_{\text{90° 회전}}
\sin\theta
$$


rotate_half(x) : (x₁, x₂) 좌표를 (−x₂, x₁) 로 바꾸는 것 = 2-D 벡터를 90° 반시계 회전.

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]      # 앞 절반
    x2 = x[..., x.shape[-1] // 2 :]      # 뒤 절반
    return torch.cat((-x2, x1), dim=-1)  # [-x2, x1]
```


apply_rotary_pos_emb(q, k, cos, sin, …) : 쿼리, 키를 “회전”시켜서 PE가 적용된 q, k를 얻게됨.

q,k	[B, H, T, D]
cos,sin	[B, T, D] (토큰별 각도 테이블)


```python
def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim:int=1)-> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]

    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

이때 필요한 cos, sin은 미리 계산해두고 필요할 때 사용함. x_new = x * cos + rotate_half(x) * sin 가 결국 $\bigl[x_{\text{even}},x_{\text{odd}}\bigr]\cos\theta + \bigl[-x_{\text{odd}},x_{\text{even}}\bigr]\sin\theta$이 되버림.

```python
class RotaryEmbedding(nn.Module):
    """
        Compute Rotary Embedding to introduce positional dependency to input sequence without additional training parameters and 
        relative distance of token position ids through angle rotation.

        Args:
            cfg: Configuration object containing:
                - lm_hidden_dim (int): Hidden dimension size.
                - lm_n_heads (int): Number of attention heads.
                - lm_re_base (float): Base for rotary embedding frequencies.
                - lm_max_position_embeddings (int): Max sequence length supported for rotary embedding.
                - lm_attn_scaling (float): Attention scaling factor.
        """
    
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads # dim of each head
        self.base = cfg.lm_re_base
        self.max_seq_len = cfg.lm_max_position_embeddings
        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        batch_size, seq_len = position_ids.shape
        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1
        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = self.inv_freq / scale
        else:
            inv_freq = self.inv_freq
            
        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flat_position_ids = position_ids.reshape(-1).float()
        
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # Reshape to include batch dimension
        freqs = freqs.reshape(batch_size, seq_len, -1)
        
        # Now create interleaved pattern
        # cos·sin 테이블도 같은 방식으로 2등분, X를 split-in-half 방식으로 나누기 때문에 이에 대응해서 같은 freqs를 붙혀야 cos, sin을 대응되게 계산할 수 있음
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin
        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        
        return cos, sin
```

* 위치 간 회전 차이가 inner-product에 **상대 위치 효과**를 반영

* **장점**: 상대 위치 정보 내재화, extrapolation 가능

* **적용 모델**: LLaMA, GPT-NeoX, ESM-2 등


### 5. ALiBi (Attention with Linear Biases)

* 위치에 따라 attention score에 **직접 벌점** 추가:

  $$
  A_{ij} \leftarrow A_{ij} - \alpha \cdot |i - j|
  $$

* **장점**: positional embedding 없이도 상대 위치 반영

* **단점**: 학습이 위치에 덜 민감, 성능 편차 있음

### 6. xPos (Sun et al., 2023)

* RoPE의 회전 각도를 **길이 무관하도록 스케일링**

* 위치 $p$에 대해:

  $$
  \text{scale}(p) = \exp(\gamma \cdot p / L)
  $$

* **장점**: 긴 시퀀스 extrapolation 성능 향상

* **단점**: RoPE보다 복잡하고 계산량 증가 가능

---

### 7. RoPE v2 (Chen et al., 2024)


* 최신 연구 – "Beyond Length Generalization: RoPE v2 for LLMs with Extrapolation and Interpolation"
* RoPE의 회전 각도에 \*\*선형 스케일 + 중심화(normalization)\*\*를 결합
* RoPE의 한계를 보완해 **길이 일반화 + 길이 간 일관성** 개선

### 주요 특징:

* 각도를 다음처럼 조절:

  $$
  \theta_{p,i}^{\text{RoPE-v2}} = \text{scale}(p) \cdot \frac{p}{10000^{2i/d}} = \underbrace{\frac{p - \mu_p}{\sigma_p}}_{\text{normalized position}} \cdot \underbrace{\frac{p}{10000^{2i/d}}}_{\text{frequency}} \cdot \lambda
  $$


* 추가로 extrapolation과 interpolation을 모두 고려한 회전

* **성능**: RoPE 대비 LLM의 길이 일반화 성능, 정확도, 추론 품질 모두 향상

* **적용**: RoPE를 쓰던 모든 구조에서 drop-in replacement 가능



# PreNorm

처음 제시된 Transformer에는 Post-Norm이 제시되었음. 이 경우 초기 러닝레이트를 작게 하고 warm-up을 길게 잡지 않으면 발산하기 쉽다고 함. Pre-Norm의 경우 warm-up 없이도 같은 러닝레이트에서 깊은 층을 쓸 수 있고 더 빠르게 수렴. 

## LayerNorm

BERT, GPT-2/3, T5, ViT에서 사용. 입력을 평균0, 분산1으로 만드는 센터링+스케일링 효과. 시퀀스 길이에 비례한 통계 계산. $(x−μ)/ \sqrt{σ^2+ε}$를 계산하다보니 편차의 평균을 반드시 구해야함. 즉, 평균과 분산 두 번 개선 필수로 하여 완벽한 센터링을 수행. ​


```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        
```

처음에 weight(gamma)=1, bias(beta)=0으로 초기화해서 학습처음에는 입력을 그냥 흘러보냄.

$$output=normalized_input×weight+bias$$

## RMSNorm

LLaMA-1/2/3, Mistral, Mixtral, Qwen에서 사용. mean=0으로 가정하고 계산을 하지 않아서 연산 절감과 메모리 사용량을 낮추면서 LayerNorm 성능 유지. (LayerNorm의 경량 버전) 입력이 0에서 벗어나면 불안정한데 PreNorm 구조로 해결. $x/ \sqrt{mean(x^2)+ε}$​ 을 계산하다보니 값 자체에 바로 평균을 내고 분산이나 평균을 빼는 (center) 연산이 없음. 

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states. 
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """
        # Compute inverse of RMS: square the tensor element-wise, mean is computed across lm_hidden_dim.
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps) # inverse of RMS
        x = x * irms * self.weight

        return x
```


# Attentions

### Causal Self attentions

attention score를 계산할 때 미래의 토큰을 무시함. 예를 들어 t번째 토큰은 t+1, t+2, ... 과 계산하지 않음. 이는 Autogressive로 텍스트 생성하는 유즈케이스에 필요함. 이전 토큰만 보고 현재 토큰에 대한 계산을 하게끔 구조적으로 강제시킴. 

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        assert cfg.n_emb % cfg.n_heads == 0, "n_emb must be divisible by n_heads"

        # Q, K, V Linear 한꺼번에 
        self.c_attn = nn.Linear(cfg.n_emb, 3 * cfg.n_emb, bias=cfg.bias)

        # Output Linear
        self.c_proj = nn.Linear(cfg.n_emb, cfg.n_emb, bias=cfg.bias)

        # regularization
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.residual_dropout = nn.Dropout(cfg.dropout)
    
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not cfg.flash:

            # lower‑triangle => 1: 현재와 과거를 봄. upper‑triangle => 0 : 미래를 보지 못하게
            # block 사이즈는 모델이 처리할 수 있는 최대길이 (미리 설정) : 1 ≤ T ≤ block_size
            self.register_buffer(
                    "bias",
                    torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size, device=device))
                )
    def forward(self, x):
        B, T, C = x.shape # (batch, length, n_emb)
        # (batch, length, n_emb) -> (batch, length, 3 * n_emb) 를 만든 뒤 3개로 나누기
        q, k, v = self.c_attn(x).split(self.cfg.n_emb, dim=2)
        k = k.view(B, T, self.cfg.n_head, C // self.cfg.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.cfg.n_head, C // self.cfg.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.cfg.n_head, C // self.cfg.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention
        if not self.flash and self.cfg.einops:
            # QK^T 계산
            attn_weights = torch.einsum("b h t d, b h s d -> b h t s", q, k) * (self.head_dim ** -0.5)

            # causal mask
            attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            # softmax 및 dtype 안정성
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
            attn_weights = self.attn_dropout(attn_weights)

            # attention 곱하기 V
            y = torch.einsum("b h t s, b h s d -> b h t d", attn_weights, v)
        elif not self.flash and self.cfg.manual_attn:
            # manual implementation of attention
            # (q @ k.transpose(-2, -1))를 하면 (B, H, T, C/H) x (B, H, C/H, T) -> (B, H, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # 1 / √dₖ로 scaled dot‑product
            mask = self.bias[:,:,:T,:T] == 0 # boolean mask가 나옴 (위쪽이 True인 삼각행렬)
            att = att.masked_fill(mask, float('-inf')) # True인 부분에 -inf를 넣음
            # softmax는 높은 값에 높은 확률을 부여
            # 아주 낮은 값이므로 0의 확률이 되게끔
            att = F.softmax(att, dim=-1) # 여전히 (B, H, T, T)
            att = self.attn_dropout(att) # (B, H, T, T)
            # matmul : A(..., m, n) × B(..., n, p) → C(..., m, p)
            y = att @ v # (B, H, T, T) x (B, H, T, C/H) -> (B, H, T, C/H)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.cfg.dropout if self.training else 0,
                is_causal=True
            )
        # (B, H, T, C/H) -> (B, T, H, C/H) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.residual_dropout(self.c_proj(y))
        return y

```

### Grouped-Query Attention (GQA)

Key, Value의 헤드를 줄임. 원래는 Q, K, V 모두 H개의 헤드를 가지고 있어서 각각 attention score가 계산되었음. $Q_{h0}$ 는 $K_{h0}$와 $Q_{h1}$는 $K_{h1}$ 이런식으로. 그런데 어차피 [B, T, HeadDim]과 [B, T, HeadDim]과의 내적은 여전히 해야함. 그런데도 도움되는 이유는 head 수를 줄이면 read out projection에서 파라미터 수가 줄어듬. 또한 KV-cache와 관련되어 있음.

![Image](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/35/7c/mqa-and-mha.component.l.ts=1744898974789.png/content/adobe-cms/us/en/think/topics/grouped-query-attention/jcr:content/root/table_of_contents/body-article-8/image_1449957912)

### KV-Cache

다음 토큰 하나를 만들기 위해 매번 새롭게 attention을 계산하면 비효율적임. 이에 대한 구현은 `https://github.dev/huggingface/nanoVLM`에 잘 나와있음. 추론할 때 사용함. 캐시 사용시 $Hkv/ Hq$으로 캐시 크기가 조정됨. (MHA의 경우 Hkv=Hq이므로 x1이다. MQA는 $1 / Hq$로 가장 이득이 많고 빠르다). KV-cache 저장/로드해서 메모리 사용량을 줄이게 된다고 함. 캐시가 줄어들면 GPU L2 ↔ HBM 왕복 데이터량이 줄어듬. FlashAttention-style 커널은 K/V를 한 번만 읽어 여러 Q에 재활용하기 때문에 메모리 대역폭 지배 구간에서 이득이 큼. 

![Image](https://camo.githubusercontent.com/6a875dea31d574c257e41ec228289fb584c5098ce6c8e09d4151e14865589a16/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f626f6e75732f6b762d63616368652f6b762d63616368652d6174746e2d312e706e673f33)

3번째 토큰에 대해 1번, 2번과 연산할 때 보면 $k^(1), v^(1), k^(2), v^(2)$는 똑같은걸 사용함.

![Image](https://camo.githubusercontent.com/cadc39c2ad8188548f897cbedd7319b1d7dc2f50563e4e5360e50c86b0b71e3d/68747470733a2f2f73656261737469616e72617363686b612e636f6d2f696d616765732f4c4c4d732d66726f6d2d736372617463682d696d616765732f626f6e75732f6b762d63616368652f6b762d63616368652d6174746e2d322e706e673f33)


```python
class LanguageModelGroupedQueryAttention(nn.Module):
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None) -> tuple[torch.Tensor, dict]:
            """
            Forward pass for grouped query attention.

            Args:
                x (Tensor): Input tensor of shape (B, T_curr, C), where
                            B = batch size,
                            T_curr = current sequence length,
                            C = embedding dimension.
                cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
                sin (Tensor): Rotary embedding sines, shape compatible with q and k.
                attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                                with 1 for tokens to attend to and 0 for padding.
                block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

            Returns:
                tuple[Tensor, dict]:
                    - Output tensor after attention and projection, shape (B, T_curr, C).
                    - Updated block_kv_cache dict for caching key-value states.
            """
            is_prefill = block_kv_cache is None

            # 만약에 generation이라면 최초 생성이 아니면 T_curr=1이 된다
            B, T_curr, C = x.size() # T_curr is the sequence length of the current input x

            q_curr = self.q_proj(x).view(B, T_curr, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T_curr, head_dim)
            k_curr = self.k_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)
            v_curr = self.v_proj(x).view(B, T_curr, self.n_kv_heads, self.head_dim).transpose(1, 2) # (B, n_kv_heads, T_curr, head_dim)

            # Apply rotary embeddings to the current q and k
            q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

            # Check if we can use cached keys and values
            # 이전에 계산해뒀던 k,v와 concat을 한뒤에 저장해둠
            if not is_prefill and block_kv_cache['key'] is not None:
                # Concatenate with cached K, V
                # k_rotated and v_curr are for the new token(s)
                k = block_kv_cache['key']
                v = block_kv_cache['value']
                k = torch.cat([k, k_rotated], dim=2)
                v = torch.cat([v, v_curr], dim=2)
                block_kv_cache['key'] = k
                block_kv_cache['value'] = v
            else: # 최초로 돌아가는 상태이면 
                # No cache, this is the first pass (prefill)
                k = k_rotated
                v = v_curr
                block_kv_cache = {'key': k, 'value': v}

            # Repeat K, V for Grouped Query Attention
            k_exp = k.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
            v_exp = v.repeat_interleave(self.n_kv_groups, dim=1) # (B, n_heads, T_kv, head_dim)
            
            T_kv = k_exp.size(2) # Total sequence length of keys/values

            # Prepare attention mask for SDPA or manual path
            # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
            additive_attn_mask = None
            if attention_mask is not None:
                # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
                # Let's make it `[B, 1, 1, T_kv]` for SDPA.
                mask_for_keys = attention_mask[:, :T_kv] # Ensure mask matches key length [B, T_kv]
                additive_attn_mask = (1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()) * torch.finfo(q.dtype).min
                # This additive_attn_mask shape is [B, 1, 1, T_kv]

            if self.sdpa and x.device.type != 'mps':
                # During decode, no additional masking needed as [1, T_kv] is naturally causal
                is_causal = (T_curr == T_kv and T_curr > 1)
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k_exp, v_exp,
                    attn_mask=additive_attn_mask, 
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal
                )
            else:
                # Manual attention implementation
                attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, T_curr, T_kv)
                # During decode: no additional masking needed as [1, T_kv] is naturally causal
                if T_curr == T_kv and T_curr > 1:
                    causal_mask_val = torch.tril(torch.ones(T_curr, T_curr, device=x.device, dtype=torch.bool)).view(1, 1, T_curr, T_curr)
                    attn = attn.masked_fill(~causal_mask_val, float('-inf'))

                if additive_attn_mask is not None: # Additive padding mask
                    # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, T_curr, T_kv]
                    attn = attn + additive_attn_mask 

                attn = F.softmax(attn, dim=-1)
                attn = self.attn_dropout(attn)
                y = attn @ v_exp
                
            y = y.transpose(1, 2).contiguous().view(B, T_curr, C)
            y = self.out_proj(y)
            y = self.resid_dropout(y)

            return y, block_kv_cache

```


https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache 여기에도 잘 나와 있음. 




### Gemma3Attention

추가된 부분. 처음 Q, K를 계산한 뒤에 RMSNorm을 걸어줌.

```python
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

query_states = self.q_norm(query_states)
key_states = self.k_norm(key_states)
```

attn_logit_softcapping 옵션 : 큰 로짓을 하드 saturate해 FP16 overflow 방어
scaling = query_pre_attn_scalar**-0.5→ 모델 파라미터로 조정 : 학습 중 Query 크기(분산)를 더 세밀히 맞춤
layer_types[layer_idx] == "sliding_attention" 인 층만 local window(sliding_window) : 긴 입력에서도 O(L·w) 메모리/연산 (w≪L)

# FeedForward Block

기본적으로 norm-linear-activation-linear-dropout-residual 형태로 구성됨. 



Mixture-of-Experts

# Activation in FF Block

## GELU

GPT2에서 사용함

```python
# Feedforward module with GELU (GPT-2)
x = linear(x)
x = gelu(x)
x = linear_projection(x)
```


## SwiGLU (Swish + Gated Linear Unit)

llama에서 사용. 게이트를 사용해서 학습을 하므로 표현력 증가됨. 기존 대비 파라미터 하나가 더 추가되지만 성능이 더 올랐다고 함. 

![Image](https://miro.medium.com/v2/resize:fit:1350/format:webp/0*BqdKcITC0ydoNriQ.png)

```python
# Feedforward module with SwiGLU (Llama 2)
x_1 = self.linear_1(x)
x_2 = self.linear_2(x)
x = silu(x_1) * x_2
x = linear_projection(x)
```

## GeGLU

![Image](https://storage.googleapis.com/lightning-avatars/litpages/01hqbsdhjzjad1g6p66ew6x37x/a54c2700-3e40-4449-b213-3e5bf68b2cca.png)

Gemma에서 사용한 activation. SwiGLU와 다른건 GELU를 사용했다는 점. 

![Image](https://storage.googleapis.com/lightning-avatars/litpages/01hqbsdhjzjad1g6p66ew6x37x/fbb7565f-8470-404c-be10-5206ac23ed64.png)

```python
# Feedforward module with GeGLU (Gemma)
x_1 = self.linear_1(x)
x_2 = self.linear_2(x)
x = gelu(x_1) * x_2
x = linear_projection(x)
```


# Encoder-Decoder 모델

최초에 제안된 Transformer가 제안된 형태는 다음과 같이 기존의 Seq2Seq 모델의 논리를 따라서 설계함. Encoder stack의 마지막 블록이 내놓는 임베딩이 의미를 잘 축약하고 효과적으로 담고 있다고 가정함. Decoder에선 이 임베딩을 받아 같은 의미를 내놓는 어떤 무언가를 내놓도록 설계함.

![Image](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

이때 Decoder는 Self-Attention 
그런데 이 구조는 언어 모델 계열에선 그렇게 유행하진 않고 있음. 이 부분에 더 exploration하는지는 아직 살펴보지 않았음.