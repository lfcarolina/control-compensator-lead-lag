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
# ===== PROJETANDO O COMPENSADOR POR AVANCO DE FASE =====
# =======================================================

print('A margem de fase é: {}'.format(MARGEM_DE_FASE))
Requisito = 35

if MARGEM_DE_FASE > Requisito:
    print('A margem de fase está dentro do pré-requisito! {} > 35'.format(MARGEM_DE_FASE))
    COMPENSADOR = 1
else:
    print('A margem de fase não está dentro do pré-requisito! {} < 35. Vamos projetar o compensador.'.format(MARGEM_DE_FASE))

    PHI_REQUERIDO = (Requisito - MARGEM_DE_FASE) + 10 #MARGEM DE SEGURANCA
    PHI_REQUERIDO_RAD = np.radians(PHI_REQUERIDO)

    ALFA = (1 - np.sin(PHI_REQUERIDO_RAD)) / (1 + np.sin(PHI_REQUERIDO_RAD))
    print('O valor de alfa é: {}'.format(ALFA))

    VALOR_BUSCA_MAGNITUDE = np.sqrt(ALFA) #PARA VALOR NEGATIVO
    PONTO_NOVA_FREQ_DESEJADA = (np.abs(mag - VALOR_BUSCA_MAGNITUDE)).argmin()

    FREQUENCIA_DESEJADA = omega[PONTO_NOVA_FREQ_DESEJADA]
    print('O valor da nova frequencia desejada é: {}'.format(FREQUENCIA_DESEJADA))

    T_AVANCO = 1 / (FREQUENCIA_DESEJADA * np.sqrt(ALFA))
    print('O valor de T (Avanço) é: {}'.format(T_AVANCO))

    GC_AVANCO = (s + 1/T_AVANCO) / (s + 1/(ALFA*T_AVANCO))
    print('Função de transferência do compensador por avanço é: {}'.format(GC_AVANCO))

    KC_COMPENSADOR = 1 / ALFA
    print('O valor de Kc é: {}'.format(KC_COMPENSADOR))

    FTMA_C_AVANCO_FINAL = KC_COMPENSADOR * GC_AVANCO * FTMA_SISTEMA_C_K_INICIAL
    print('A função de transferência da planta compensada final é: {}'.format(FTMA_C_AVANCO_FINAL))


# ========================================================
# ===== PLOTANDO BODE E NYQUIST PARA "FTMA_C_AVANCO" =====
# ========================================================

plt.figure()
ct.bode_plot(FTMA_C_AVANCO_FINAL, dB=True, margins=True)
plt.suptitle("Bode do Sistema Compensado")
plt.show()

fig, ax = plt.subplots()
ct.nyquist_plot(FTMA_C_AVANCO_FINAL, ax=ax)
ax.set_title("Nyquist do Sistema Compensado")
plt.show()

# ====================================================================
# ===== PLOTANDO RESPOSTA AO DEGRAU E RAMPA PARA "FTMA_C_AVANCO" =====
# ====================================================================

FTMF_C_AVANCO_FINAL = ct.feedback(FTMA_C_AVANCO_FINAL)

print('A função de transferência de malha fechada da planta com compensador por avanço é: {}'.format(FTMF_C_AVANCO_FINAL))

t1, y1 = ct.step_response(FTMF_C_AVANCO_FINAL)
plt.figure()
plt.plot(t1, y1)
plt.title("Resposta ao degrau do sistema compensado por avanço")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

t2 = np.linspace(0, 10, 1000)
rampa = t2
t3, y3 = ct.forced_response(FTMF_C_AVANCO_FINAL, T=t2, U=rampa)

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

mag, phase, omega = ct.bode(FTMA_C_AVANCO_FINAL, plot=False)
PONTO_MARGEM_FASE_FINAL = (np.abs(mag - 1)).argmin()
MARGEM_DE_FASE_FINAL = phase[PONTO_MARGEM_FASE_FINAL] * 180/np.pi + 180
print('A margem de fase *final* é: {}'.format(MARGEM_DE_FASE_FINAL))

GANHO_FINAL = ct.dcgain(ct.minreal(s * FTMA_C_AVANCO_FINAL))
print('O ganho final é: {}'.format(GANHO_FINAL))

ERRO_FINAL = 1 / GANHO_FINAL
print('O erro estatico final é: {}'.format(ERRO_FINAL))

if MARGEM_DE_FASE_FINAL > 35 and ERRO_FINAL <= 0.050001: #Para desprezar as imperfeições.
    print('Todos os requisitos foram atendidos!')
else:
    print('ERRO!')
