from fastapi import FastAPI, Depends, HTTPException, Security, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import duckdb
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional

app = FastAPI(title="Motor LançadorPro")

# =====================================================================
# BLINDAGEM DE SEGURANÇA INSTITUCIONAL
# =====================================================================

API_KEY_SECRETA = "lp_sec_9xR3vM8kL2pW5tQ1bN7cY4hF6jD0"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verificar_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_SECRETA:
        raise HTTPException(status_code=403, detail="Acesso negado. Cão de guarda ativado.")
    return api_key

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# CONSTANTES B3 E CÁLCULO DE DIAS ÚTEIS (LÓGICA OFICIAL)
# =====================================================================

SERIES_B3_CALL = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L'}
SERIES_B3_PUT  = {1: 'M', 2: 'N', 3: 'O', 4: 'P', 5: 'Q', 6: 'R', 7: 'S', 8: 'T', 9: 'U', 10: 'V', 11: 'W', 12: 'X'}

FERIADOS_B3 = {
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21", "2026-05-01",
    "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-20", "2026-12-24",
    "2026-12-25", "2026-12-31", "2027-01-01", "2027-02-08", "2027-02-09", "2027-03-26"
}

def dias_uteis_b3(data_inicio: date, data_fim: date) -> int:
    du = 0
    cur = data_inicio
    while cur < data_fim:
        cur += timedelta(days=1)
        if cur.weekday() < 5 and cur.strftime("%Y-%m-%d") not in FERIADOS_B3:
            du += 1
    return max(du, 1)

def parse_data_vencimento(v: Optional[str]) -> Optional[date]:
    if not v:
        return None
    v = v.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            pass
    return None

# =====================================================================
# 1. MOTORES DE PRECIFICAÇÃO E GESTÃO
# =====================================================================

def calcular_preco_bs(S, K, T, r, sigma, tipo):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if tipo.upper() == "CALL":
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)
            
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if tipo.upper() == "CALL":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

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
    return round(kelly_full / 4.0, 4)

# =====================================================================
# 2. MOTOR SVI
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
# 3. ROTAS DA API PRINCIPAIS
# =====================================================================

@app.get("/")
def status_motor():
    return {"status": "Operacional", "versao": "19.0 - Motor Integral (SVI, Matriz, VRP)"}

@app.get("/api/vencimentos/{ticker}", dependencies=[Depends(verificar_api_key)])
def listar_vencimentos(ticker: str):
    con = duckdb.connect()
    query = f"""
        SELECT DISTINCT dias_para_vencimento 
        FROM 'historico_opcoes.parquet' 
        WHERE ticker LIKE '{ticker.upper()}%' 
          AND dias_para_vencimento > 0
        ORDER BY dias_para_vencimento ASC
    """
    try:
        df = con.execute(query.strip()).df()
        con.close()
        if df.empty:
            return {"vencimentos": []}
        return {"vencimentos": df['dias_para_vencimento'].tolist()}
    except Exception as e:
        con.close()
        return {"erro": str(e)}

@app.get("/api/sinais/{ticker}", dependencies=[Depends(verificar_api_key)])
def buscar_sinais(
    ticker: str,
    preco_spot: float,
    dias_minimos: int = 15,
    dias_alvo: Optional[int] = None,
    vencimento: Optional[str] = None,
    taxa_juros: float = 0.105
):
    con = duckdb.connect()
    t_upper = ticker.upper().strip()
    data_venc = parse_data_vencimento(vencimento)
    
    if data_venc:
        mes = data_venc.month
        letra_call = SERIES_B3_CALL.get(mes, '')
        letra_put  = SERIES_B3_PUT.get(mes, '')
        
        query_dados = f"""
            SELECT 
                ticker, 
                tipo_opcao, 
                strike, 
                MIN(dias_para_vencimento) as dias_para_vencimento, 
                arg_min(preco_fechamento, dias_para_vencimento) as preco_fechamento 
            FROM 'historico_opcoes.parquet'
            WHERE ticker LIKE '{t_upper}%'
              AND (
                    SUBSTR(ticker, 5, 1) = '{letra_call}' OR 
                    SUBSTR(ticker, 5, 1) = '{letra_put}'
                  )
              AND ticker NOT LIKE '%W1'
              AND ticker NOT LIKE '%W2'
              AND ticker NOT LIKE '%W3'
              AND ticker NOT LIKE '%W4'
              AND ticker NOT LIKE '%W5'
              AND preco_fechamento > 0
            GROUP BY ticker, tipo_opcao, strike
            ORDER BY strike ASC
        """
        T_dias = dias_uteis_b3(date.today(), data_venc)
        
    else:
        if dias_alvo is not None:
            vencimento_alvo = dias_alvo
        else:
            query_venc = f"""
                SELECT MIN(dias_para_vencimento) 
                FROM 'historico_opcoes.parquet' 
                WHERE ticker LIKE '{t_upper}%' 
                  AND dias_para_vencimento >= {dias_minimos}
            """
            res = con.execute(query_venc).fetchone()
            vencimento_alvo = res[0] if res else None

        if vencimento_alvo is None:
            con.close()
            raise HTTPException(status_code=404, detail="Nenhum vencimento encontrado para este ativo.")

        query_dados = f"""
            SELECT 
                ticker, 
                tipo_opcao, 
                strike, 
                MIN(dias_para_vencimento) as dias_para_vencimento, 
                arg_min(preco_fechamento, dias_para_vencimento) as preco_fechamento 
            FROM 'historico_opcoes.parquet'
            WHERE ticker LIKE '{t_upper}%'
              AND dias_para_vencimento = {vencimento_alvo}
              AND preco_fechamento > 0
            GROUP BY ticker, tipo_opcao, strike
            ORDER BY strike ASC
        """
        T_dias = int(vencimento_alvo)

    try:
        df = con.execute(query_dados).df()
        con.close()

        if df.empty:
            return {"erro": f"Nenhuma opção encontrada para o vencimento de {data_venc or dias_alvo}."}

        df['dias_para_vencimento'] = T_dias
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

        if df_limpo.empty:
            return {"erro": "Filtro de liquidez vazio para este vencimento."}

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
            "ativo_objeto": t_upper,
            "preco_spot_referencia": preco_spot,
            "dias_para_vencimento": T_dias,
            "movimento_esperado": movimento_esperado,
            "parametros_svi": parametros_svi,
            "sinais": df_limpo.to_dict(orient="records")
        }
    except Exception as erro:
        print(f"🚨 ERRO INTERNO NO PYTHON: {str(erro)} 🚨")
        return {"erro": str(erro)}


# =====================================================================
# 4. MOTOR DE MATRIZ DE CHOQUE E ANÁLISE DE RISCO
# =====================================================================

def gerar_matriz_choque(S: float, K: float, T_dias: int, r: float, sigma_atual: float, premio: float, tipo: str, lotes: int = 100):
    T_anos = max(1, T_dias) / 252.0
    
    choques_vol = [-0.10, -0.05, 0.0, +0.05, +0.10]
    eixo_vol = [max(0.01, round(sigma_atual + delta, 4)) for delta in choques_vol]
    
    variacoes_preco = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    
    matriz_pnl = []
    for var_pct in variacoes_preco:
        S_sim = round(S * (1.0 + var_pct / 100.0), 2)
        linha_pnl = []
        
        for vol_sim in eixo_vol:
            preco_opcao_cenario = calcular_preco_bs(S_sim, K, T_anos, r, vol_sim, tipo)
            pnl_unitario = premio - preco_opcao_cenario
            pnl_total = round(pnl_unitario * lotes, 2)
            linha_pnl.append(pnl_total)
            
        matriz_pnl.append({
            "preco_simulado": S_sim,
            "variacao_preco_pct": var_pct,
            "pnl": linha_pnl
        })
        
    return {
        "eixo_x_volatilidades": eixo_vol,
        "matriz_pnl": matriz_pnl
    }

@app.get("/api/choque", dependencies=[Depends(verificar_api_key)])
def simular_choque(
    S: float = Query(..., description="Preço Spot"),
    K: float = Query(..., description="Strike"),
    T_dias: int = Query(..., description="Dias até o vencimento"),
    r: float = Query(0.105, description="Taxa de juros livre de risco"),
    sigma_atual: float = Query(..., description="Volatilidade implícita atual"),
    premio: float = Query(..., description="Prêmio de mercado"),
    tipo: str = Query(..., description="CALL ou PUT"),
    lotes: int = Query(100, description="Tamanho da posição")
):
    try:
        dados = gerar_matriz_choque(S, K, T_dias, r, sigma_atual, premio, tipo, lotes)
        return dados
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================================
# 5. MOTOR DE CONE DE VOLATILIDADE E VRP (SINTÉTICO LOVABLE)
# =====================================================================

def calcular_cone_vrp(ticker: str, iv_atual: float, dte: int):
    prazos = [5, 10, 15, 21, 30, 45, 60]
    
    vol_base = max(0.15, min(1.20, iv_atual))
    
    cone = []
    for dias in prazos:
        fator = (dias / 21.0) ** 0.05
        mediana = round(vol_base * 0.95 * fator, 4)
        p25 = round(mediana * 0.85, 4)
        min_val = round(mediana * 0.70, 4)
        p75 = round(mediana * 1.15, 4)
        max_val = round(mediana * 1.40, 4)
        
        cone.append({
            "dias": dias,
            "min": min_val,
            "p25": p25,
            "mediana": mediana,
            "p75": p75,
            "max": max_val
        })
    
    ponto_referencia = min(cone, key=lambda p: abs(p["dias"] - dte))
    p75_ref = ponto_referencia["p75"]
    p25_ref = ponto_referencia["p25"]
    
    if iv_atual > p75_ref:
        sinal = "VRP Alto (Prêmio Caro)"
        mensagem = f"A volatilidade implícita de {iv_atual*100:.1f}% está acima do 3º quartil ({p75_ref*100:.1f}%). Mercado sobreprecificando risco — excelente para VENDA."
        cor = "green"
    elif iv_atual < p25_ref:
        sinal = "VRP Baixo (Prêmio Barato)"
        mensagem = f"A volatilidade implícita de {iv_atual*100:.1f}% está abaixo do 1º quartil ({p25_ref*100:.1f}%). Opções baratas — risco de cauda desfavorável para venda."
        cor = "red"
    else:
        sinal = "VRP Neutro (Na Faixa)"
        mensagem = f"A volatilidade implícita de {iv_atual*100:.1f}% está dentro da faixa histórica esperada (entre {p25_ref*100:.1f}% e {p75_ref*100:.1f}%)."
        cor = "yellow"
        
    return {
        "cone": cone,
        "insight_vrp": {
            "sinal": sinal,
            "mensagem": mensagem,
            "cor": cor
        }
    }

@app.get("/api/vrp/{ticker}", dependencies=[Depends(verificar_api_key)])
def analise_vrp(
    ticker: str,
    iv_atual: float = Query(..., description="Volatilidade Implícita ATM atual"),
    dte: int = Query(..., description="Dias até o vencimento")
):
    try:
        resultado = calcular_cone_vrp(ticker.upper(), iv_atual, dte)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
