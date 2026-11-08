from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import duckdb
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd

app = FastAPI(title="Motor LançadorPro")

# =====================================================================
# BLINDAGEM DE SEGURANÇA INSTITUCIONAL
# =====================================================================

# 1. Definição da Chave Secreta de Alta Entropia
API_KEY_SECRETA = "lp_sec_9xR3vM8kL2pW5tQ1bN7cY4hF6jD0"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verificar_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_SECRETA:
        raise HTTPException(status_code=403, detail="Acesso negado. Cão de guarda ativado.")
    return api_key

# 2. CORS Liberado com Suporte a Todos os Ambientes (Preview, Localhost e Produção)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# 1. MOTORES DE PRECIFICAÇÃO, VOLATILIDADE E GESTÃO (BS, BINOMIAL, POP, KELLY)
# =====================================================================

def calcular_iv_bs_newton(preco_mercado, S, K, T, r, tipo_opcao):
    if T <= 0 or preco_mercado <= 0: return 0.0
    sigma = 0.5  
    for _ in range(100): 
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if tipo_opcao == 'CALL':
            preco_teorico = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            preco_teorico = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-8: break
            
        diff = preco_teorico - preco_mercado
        if abs(diff) < 1e-4: break
            
        sigma = sigma - (diff / vega)
        if sigma <= 0.001: return 0.001
    return round(sigma, 4)

def calcular_preco_binomial_americana(S, K, T, r, sigma, N=50, tipo='CALL'):
    if T <= 0 or sigma <= 0: return 0.0
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    
    precos_ativos = np.array([S * (u ** (N - i)) * (d ** i) for i in range(N + 1)])
    
    if tipo == 'CALL':
        valores = np.maximum(precos_ativos - K, 0)
    else:
        valores = np.maximum(K - precos_ativos, 0)
        
    fator_desconto = math.exp(-r * dt)
    for j in range(N - 1, -1, -1):
        precos_ativos = precos_ativos[:-1] / u 
        valores_esperados = fator_desconto * (p * valores[:-1] + (1 - p) * valores[1:])
        
        if tipo == 'CALL':
            valores_intrinsecos = np.maximum(precos_ativos - K, 0)
        else:
            valores_intrinsecos = np.maximum(K - precos_ativos, 0)
            
        valores = np.maximum(valores_esperados, valores_intrinsecos)
        
    return valores[0]

def calcular_iv_binomial_bissecao(preco_mercado, S, K, T, r, tipo='CALL'):
    if T <= 0 or preco_mercado <= 0: return 0.0
    sigma_baixo = 0.001
    sigma_alto = 3.0 
    
    preco_baixo = calcular_preco_binomial_americana(S, K, T, r, sigma_baixo, N=40, tipo=tipo)
    preco_alto = calcular_preco_binomial_americana(S, K, T, r, sigma_alto, N=40, tipo=tipo)
    
    if preco_mercado < preco_baixo: return 0.001
    if preco_mercado > preco_alto: return 3.0
    
    for _ in range(30):
        sigma_mid = (sigma_baixo + sigma_alto) / 2
        preco_mid = calcular_preco_binomial_americana(S, K, T, r, sigma_mid, N=40, tipo=tipo)
        
        diff = preco_mid - preco_mercado
        if abs(diff) < 1e-3: break
            
        if preco_mid > preco_mercado:
            sigma_alto = sigma_mid
        else:
            sigma_baixo = sigma_mid
            
    return round(sigma_mid, 4)

def calcular_pop(S, K, premio, T, r, sigma, tipo_opcao):
    if T <= 0 or sigma <= 0 or premio <= 0: return 0.0
    
    if tipo_opcao == 'CALL':
        be = K + premio
        d2 = (math.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        pop = norm.cdf(-d2)
    else:
        be = K - premio
        if be <= 0: return 1.0 
        d2 = (math.log(S / be) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        pop = norm.cdf(d2)
        
    return round(pop, 4)

def calcular_kelly_otm(pop: float, multiplicador_stop_loss: float = 3.0):
    if pop <= 0 or pop >= 1: return 0.0
    
    p = pop
    q = 1.0 - p
    perda_liquida = multiplicador_stop_loss - 1.0
    
    if perda_liquida <= 0: return 0.0
    
    b = 1.0 / perda_liquida
    kelly_full = p - (q / b)
    
    if kelly_full <= 0: return 0.0
    
    kelly_fracionado = kelly_full / 4.0
    return round(kelly_fracionado, 4)

# =====================================================================
# 2. MOTOR SVI (STOCHASTIC VOLATILITY INSPIRED)
# =====================================================================

def svi_variancia_total(k, params):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def verificar_durrleman(k_array, params):
    dk = 1e-4
    w = svi_variancia_total(k_array, params)
    w_plus = svi_variancia_total(k_array + dk, params)
    w_minus = svi_variancia_total(k_array - dk, params)
    dw = (w_plus - w_minus) / (2 * dk)
    d2w = (w_plus - 2 * w + w_minus) / (dk**2)
    g_k = (1 - (k_array * dw) / (2 * w))**2 - (dw**2 / 4) * (1 / w + 0.25) + (d2w / 2)
    return np.all(g_k >= 0) 

def calibrar_svi_superficie(df_opcoes, S, T_dias, r):
    if df_opcoes.empty: return None
    
    T = T_dias / 252.0
    F = S * math.exp(r * T) 
    k_target = np.log(df_opcoes['strike'] / F).values
    w_target = (df_opcoes['vol_implicita']**2 * T).values
    params_iniciais = [0.04, 0.1, -0.5, 0.0, 0.1]
    bounds = ((1e-5, max(w_target)), (1e-5, 1.0), (-0.99, 0.99), (-2.0, 2.0), (0.01, 1.0))
    
    def funcao_objetivo(params):
        w_calc = svi_variancia_total(k_target, params)
        erro = np.sum((w_calc - w_target)**2)
        if not verificar_durrleman(k_target, params): erro += 1e6 
        return erro

    res = minimize(funcao_objetivo, params_iniciais, method='L-BFGS-B', bounds=bounds)
    if res.success:
        return {
            "a": round(res.x[0], 6), "b": round(res.x[1], 6),
            "rho": round(res.x[2], 6), "m": round(res.x[3], 6),
            "sigma": round(res.x[4], 6),
            "livre_arbitragem_durrleman": bool(verificar_durrleman(k_target, res.x))
        }
    return None

# =====================================================================
