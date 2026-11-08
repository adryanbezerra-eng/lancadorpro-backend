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

# 2. Fechamento do CORS (Lista de Convidados)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",           
        "https://lancadorpro.lovable.app", 
        "https://lancadorpro.app",         # <--- ADICIONE ESTA LINHA COM O SEU DOMÍNIO OFICIAL
        "https://www.lancadorpro.app",     # <--- ADICIONE ESTA TAMBÉM POR GARANTIA
    ],
    allow_credentials=True,
    allow_methods=["GET"],                 
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
# 3. MOTOR DE CONE DE VOLATILIDADE (VRP)
# =====================================================================

def calcular_cone_volatilidade(ticker: str, dias_historico: int = 504):
    con = duckdb.connect()
    query = f"""
        SELECT data, preco_fechamento 
        FROM 'historico_acoes.parquet' 
        WHERE ticker LIKE '{ticker.upper()}%' 
        ORDER BY data ASC 
        LIMIT {dias_historico}
    """
    try:
        df = con.execute(query).fetchdf()
        if df.empty:
            return {"erro": "Histórico insuficiente."}
            
        df['log_return'] = np.log(df['preco_fechamento'] / df['preco_fechamento'].shift(1))
        janelas = [20, 40, 60, 120]
        cone = []
        
        for janela in janelas:
            df[f'vol_{janela}'] = df['log_return'].rolling(window=janela).std() * math.sqrt(252)
            vols = df[f'vol_{janela}'].dropna()
            if not vols.empty:
                cone.append({
                    "dias": janela, "min": round(vols.min(), 4),
                    "p25": round(np.percentile(vols, 25), 4), "mediana": round(np.percentile(vols, 50), 4),
                    "p75": round(np.percentile(vols, 75), 4), "max": round(vols.max(), 4)
                })
        return cone
    except Exception as e:
        return {"erro": str(e)}

def avaliar_vrp(iv_atual: float, dte: int, cone: list):
    if not cone or "erro" in cone:
        return {"sinal": "ERRO", "cor": "gray", "mensagem": "Sem dados."}
        
    janela_alvo = min(cone, key=lambda x: abs(x['dias'] - dte))
    vrp_spread = iv_atual - janela_alvo['mediana']
    
    if iv_atual >= janela_alvo['p75']:
        return {"sinal": "VENDA FORTE", "cor": "green", "mensagem": f"VRP Esticado (+{vrp_spread:.2%})"}
    elif iv_atual <= janela_alvo['p25']:
        return {"sinal": "COMPRA", "cor": "red", "mensagem": f"VRP Negativo ({vrp_spread:.2%})"}
    return {"sinal": "NEUTRO", "cor": "yellow", "mensagem": "Margem normal"}

# =====================================================================
# 4. MOTOR DE MATRIZ DE ESTRESSE (HEATMAP 2D)
# =====================================================================

def gerar_matriz_choque(S, K, T_dias, r, sigma_atual, premio_recebido, tipo_opcao, qte_lotes=100):
    if T_dias <= 0 or sigma_atual <= 0: return {"erro": "Parâmetros inválidos."}
    T_anos = max(1, T_dias - 1) / 252.0
    choques_preco = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    choques_vol = [-0.20, 0.0, 0.20, 0.40, 0.60]
    matriz = []
    
    for choque_p in choques_preco:
        linha = {"variacao_preco_pct": choque_p, "preco_simulado": round(S * (1 + choque_p), 2)}
        cenarios_vol = {}
        for choque_v in choques_vol:
            iv_simulada = max(0.01, sigma_atual + choque_v)
            S_simulado = S * (1 + choque_p)
            
            d1 = (math.log(S_simulado / K) + (r + 0.5 * iv_simulada**2) * T_anos) / (iv_simulada * math.sqrt(T_anos))
            d2 = d1 - iv_simulada * math.sqrt(T_anos)
            
            if tipo_opcao == 'CALL':
                preco_estresse = S_simulado * norm.cdf(d1) - K * math.exp(-r * T_anos) * norm.cdf(d2)
            else:
                preco_estresse = K * math.exp(-r * T_anos) * norm.cdf(-d2) - S_simulado * norm.cdf(-d1)
                
            pnl = (premio_recebido - preco_estresse) * qte_lotes
            cenarios_vol[f"iv_{int((sigma_atual + choque_v)*100)}"] = round(pnl, 2)
            
        linha["cenarios_volatilidade"] = cenarios_vol
        matriz.append(linha)
        
    return {
        "eixo_x_volatilidades": [round(sigma_atual + v, 4) for v in choques_vol],
        "eixo_y_precos": [round(S * (1 + p), 2) for p in choques_preco],
        "matriz_pnl": matriz,
        "premio_entrada_total": round(premio_recebido * qte_lotes, 2)
    }

# =====================================================================
# 5. ROTAS DA API BLINDADAS
# =====================================================================

@app.get("/")
def status_motor():
    return {"status": "Operacional", "versao": "11.0 - API Totalmente Blindada (CORS + Chave)"}

@app.get("/api/sinais/{ticker}", dependencies=[Depends(verificar_api_key)])
def buscar_sinais(ticker: str, preco_spot: float, dias_minimos: int = 15, taxa_juros: float = 0.105):
    con = duckdb.connect()
    
    query = f"""
        SELECT ticker, tipo_opcao, strike, dias_para_vencimento, preco_fechamento 
        FROM 'historico_opcoes.parquet'
        WHERE ticker LIKE '{ticker.upper()}%'
          AND dias_para_vencimento = (
              SELECT MIN(dias_para_vencimento) 
              FROM 'historico_opcoes.parquet' 
              WHERE ticker LIKE '{ticker.upper()}%' AND dias_para_vencimento >= {dias_minimos}
          )
          AND preco_fechamento > 0
        ORDER BY strike ASC
    """
    
    try:
        df = con.execute(query).fetchdf()
        if df.empty: return {"erro": "Nenhuma opção encontrada."}
        
        T_dias = int(df['dias_para_vencimento'].iloc[0])
        T_anos = T_dias / 252.0
        
        iv_calculada = []
        pop_calculada = []
        be_calculado = []
        kelly_calculado = [] 
        
        for index, row in df.iterrows():
            premio = row['preco_fechamento']
            if row['tipo_opcao'] == 'CALL':
                iv = calcular_iv_binomial_bissecao(premio, preco_spot, row['strike'], T_anos, taxa_juros, 'CALL')
                be = row['strike'] + premio
            else:
                iv = calcular_iv_bs_newton(premio, preco_spot, row['strike'], T_anos, taxa_juros, 'PUT')
                be = row['strike'] - premio
                
            iv_calculada.append(iv)
            be_calculado.append(round(be, 2))
            
            pop_atual = calcular_pop(preco_spot, row['strike'], premio, T_anos, taxa_juros, iv, row['tipo_opcao'])
            pop_calculada.append(pop_atual)
            kelly_calculado.append(calcular_kelly_otm(pop_atual, 3.0))
            
        df['vol_implicita'] = iv_calculada
        df['break_even'] = be_calculado
        df['pop'] = pop_calculada
        df['kelly_frac'] = kelly_calculado 
        
        df['moneyness'] = df['strike'] / preco_spot
        df_limpo = df[
            (df['vol_implicita'] > 0.01) & 
            (df['vol_implicita'] < 2.50) & 
            (df['moneyness'] > 0.70) & 
            (df['moneyness'] < 1.30)
        ].copy()
        
        if df_limpo.empty: return {"erro": "Filtro de liquidez vazio."}
        
        df_otm = df_limpo[
            ((df_limpo['tipo_opcao'] == 'CALL') & (df_limpo['strike'] >= preco_spot)) |
            ((df_limpo['tipo_opcao'] == 'PUT') & (df_limpo['strike'] <= preco_spot))
        ].copy()
        
        parametros_svi = calibrar_svi_superficie(df_otm, S=preco_spot, T_dias=T_dias, r=taxa_juros)
        df_limpo = df_limpo.drop(columns=['moneyness'])
        
        idx_atm = (df_limpo['strike'] - preco_spot).abs().idxmin()
        iv_atm = df_limpo.loc[idx_atm, 'vol_implicita']
        em_valor = preco_spot * iv_atm * math.sqrt(T_anos)
        movimento_esperado = {
            "valor_absoluto": round(em_valor, 2),
            "limite_inferior": round(preco_spot - em_valor, 2),
            "limite_superior": round(preco_spot + em_valor, 2),
            "iv_atm_utilizada": round(iv_atm, 4)
        }
        
        return {
            "ativo_objeto": ticker.upper(),
            "preco_spot_referencia": preco_spot,
            "dias_para_vencimento": T_dias,
            "movimento_esperado": movimento_esperado,
            "parametros_svi": parametros_svi,
            "sinais": df_limpo.to_dict(orient="records")
        }
    except Exception as erro:
        return {"erro": str(erro)}

@app.get("/api/vrp/{ticker}", dependencies=[Depends(verificar_api_key)])
def analise_vrp(ticker: str, iv_atual: float, dte: int):
    cone_data = calcular_cone_volatilidade(ticker)
    if isinstance(cone_data, dict) and "erro" in cone_data: return cone_data
    alerta = avaliar_vrp(iv_atual, dte, cone_data)
    return {"ativo": ticker.upper(), "iv_atual": iv_atual, "dte": dte, "cone": cone_data, "insight_vrp": alerta}

@app.get("/api/choque", dependencies=[Depends(verificar_api_key)])
def simular_choque_opcao(S: float, K: float, T_dias: int, r: float, sigma_atual: float, premio: float, tipo: str, lotes: int = 100):
    try:
        return gerar_matriz_choque(S, K, T_dias, r, sigma_atual, premio, tipo.upper(), lotes)
    except Exception as e:
        return {"erro": str(e)}
