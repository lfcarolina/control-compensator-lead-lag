import warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

import numpy as np
import control as ct
import matplotlib.pyplot as plt
from control.grid import sin

# Objetivo: projetar dois tipos de compensadores, por atraso de fase e avanço de fase, para que o
#erro em regime permanente para uma entrada em degrau seja menor ou igual a 0,05 com uma
#margem de fase maior do que 35 graus.

# ========================================
# ========= SORTEANDO OS NÚMEROS =========
# ========================================

K1=np.random.uniform(4, 10)
P1=np.random.uniform(1, 3)

s = ct.TransferFunction.s
FTMA_SISTEMA = K1 / (s * (s + P1))
print('A função de transferência da planta é: {}' .format(FTMA_SISTEMA));

# =============================================================
# ==== CALCULANDO O GANHO NECESSÁRIO PARA O ERRO REQUERIDO ====
# =============================================================

K_INICIAL = 20 * (P1/K1)
print('O valor de Kc, a fim de atender a especificação do projeto é: {}'.format(K_INICIAL))

# ===========================================================================
# = CALCULANDO MARGEM DE GANHO E MARGEM DE FASE PARA VERIFICAR ESTABILIDADE =
# ===========================================================================

FTMA_SISTEMA_C_K_INICIAL = K_INICIAL * K1 / (s * (s + P1))
print('A função de transferência da planta é: {}' .format(FTMA_SISTEMA_C_K_INICIAL));

mag, phase, omega = ct.bode(FTMA_SISTEMA_C_K_INICIAL, plot=False)

PONTO_MARGEM_FASE = (np.abs(mag - 1)).argmin()
MARGEM_DE_FASE = phase[PONTO_MARGEM_FASE] * 180/np.pi + 180
print('A margem de fase é: {}'.format(MARGEM_DE_FASE))

PONTO_MARGEM_GANHO = np.argmin(np.abs(phase*180/np.pi + 180))
MARGEM_DE_GANHO = 1 / mag[PONTO_MARGEM_GANHO]
MARGEM_DE_GANHO_DB = 20 * np.log10(MARGEM_DE_GANHO)
print('A margem de ganho é: {}'.format(MARGEM_DE_GANHO_DB))

if MARGEM_DE_GANHO_DB > 0 and MARGEM_DE_FASE > 0:
    print('O sistema é estável!')
else:
    print('O sistema não é estável!')

# ====================================================================
# === PLOTANDO BODE E NYQUIST PARA "FTMA_SISTEMA" COM O GANHO "Kc" ===
# ====================================================================

ct.bode_plot(FTMA_SISTEMA_C_K_INICIAL, dB=True, margins=True)
plt.suptitle("Diagrama de Bode")
plt.show()

fig, ax = plt.subplots()
ct.nyquist_plot(FTMA_SISTEMA_C_K_INICIAL, ax=ax)
ax.set_title("Diagrama de Nyquist")
plt.show()

# =======================================================
# ===== PROJETANDO O COMPENSADOR POR ATRASO DE FASE =====
# =======================================================

print('A margem de fase é: {}'.format(MARGEM_DE_FASE))
Requisito = 35

if MARGEM_DE_FASE > Requisito:
    print('A margem de fase está dentro do pré-requisito! {} > 35'.format(MARGEM_DE_FASE))
    COMPENSADOR = 1
else:
    print('A margem de fase não está dentro do pré-requisito! {} < 35. Vamos projetar o compensador.'.format(MARGEM_DE_FASE))

    PHI_REQUERIDO = (Requisito) - 180 + 10 #MARGEM DE SEGURANCA

    phase_deg = phase * (180.0 / np.pi)

    INDICE = np.argmin(np.abs(phase_deg - PHI_REQUERIDO))
    FREQUENCIA_REQUERIDA = omega[INDICE]
    print('A frequencia desejada é: {}'.format(FREQUENCIA_REQUERIDA))

    T_ATRASO = 10 / (FREQUENCIA_REQUERIDA)
    print('O valor de T (Atraso) é: {}'.format(T_ATRASO))

    BETA = mag[INDICE]
    print('O valor de BETA é: {}'.format(BETA))

    GC_ATRASO = (s + 1/T_ATRASO) / (s + 1/(BETA*T_ATRASO))
    print('Função de transferência do compensador por atraso é: {}'.format(GC_ATRASO))

    KC_COMPENSADOR = K_INICIAL / BETA
    print('O valor de Kc é: {}'.format(KC_COMPENSADOR))

    FTMA_C_ATRASO = KC_COMPENSADOR * GC_ATRASO * FTMA_SISTEMA
    print('A função de transferência da planta compensada é: {}'.format(FTMA_C_ATRASO))

# ========================================================
# ===== PLOTANDO BODE E NYQUIST PARA "FTMA_C_ATRASO" =====
# ========================================================

plt.figure()
ct.bode_plot(FTMA_C_ATRASO, dB=True, margins=True)
plt.suptitle("Bode do Sistema Compensado")
plt.show()

fig, ax = plt.subplots()
ct.nyquist_plot(FTMA_C_ATRASO, ax=ax)
ax.set_title("Nyquist do Sistema Compensado")
plt.show()

# ====================================================================
# ===== PLOTANDO RESPOSTA AO DEGRAU E RAMPA PARA "FTMA_C_ATRASO" =====
# ====================================================================

FTMF_C_ATRASO = ct.feedback(FTMA_C_ATRASO)

print('A função de transferência de malha fechada da planta com compensador por avanço é: {}'.format(FTMF_C_ATRASO))

t1, y1 = ct.step_response(FTMF_C_ATRASO)
plt.figure()
plt.plot(t1, y1)
plt.title("Resposta ao degrau do sistema compensado por atraso")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

t2 = np.linspace(0, 10, 1000)
rampa = t2
t3, y3 = ct.forced_response(FTMF_C_ATRASO, T=t2, U=rampa)

plt.figure()
plt.plot(t3, y3, label="Saída")
plt.plot(t2, rampa, '--', label="Entrada Rampa")
plt.title("Resposta à Rampa do Sistema Compensado")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# ============================================
# ===== CONFERINDO REQUISTOS SOLICITADOS =====
# ============================================

mag, phase, omega = ct.bode(FTMA_C_ATRASO, plot=False)
PONTO_MARGEM_FASE_FINAL = (np.abs(mag - 1)).argmin()
MARGEM_DE_FASE_FINAL = phase[PONTO_MARGEM_FASE_FINAL] * 180/np.pi + 180
print('A margem de fase *final* é: {}'.format(MARGEM_DE_FASE_FINAL))

GANHO_FINAL = ct.dcgain(ct.minreal(s * FTMA_C_ATRASO))
print('O ganho final é: {}'.format(GANHO_FINAL))

ERRO_FINAL = 1 / GANHO_FINAL
print('O erro estatico final é: {}'.format(ERRO_FINAL))

if MARGEM_DE_FASE_FINAL > 35 and ERRO_FINAL <= 0.050001: #Para desprezar as imperfeições.
    print('Todos os requisitos foram atendidos!')
else:
    print('ERRO!')
