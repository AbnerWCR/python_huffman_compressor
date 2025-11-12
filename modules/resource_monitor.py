import time
import psutil
import threading
import functools
import os
import pandas as pd
from typing import Callable, Any
import csv

# Obtém o processo atual. Fazemos isso globalmente para que a thread
# de monitoramento e a função principal estejam olhando para o mesmo processo.
CURRENT_PROCESS = psutil.Process(os.getpid())

class ResourceMonitor(threading.Thread):
    """
    Uma thread que monitora o uso de CPU e Memória (RSS) em intervalos
    regulares até que seja sinalizada para parar.
    """
    def __init__(self, poll_interval: float = 0.1):
        super().__init__(daemon=True)
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()
        self.max_cpu_percent = 0.0
        self.max_mem_rss = 0
        # A primeira chamada a cpu_percent() sem intervalo retorna 0.
        # Chamamos uma vez para "preparar" o monitoramento.
        CURRENT_PROCESS.cpu_percent(interval=None) 

    def run(self):
        """Inicia o loop de monitoramento."""
        while not self.stop_event.is_set():
            try:
                # 1. Uso de CPU
                # Obtém o uso de CPU desde a última chamada ou no intervalo.
                # Nota: Em um sistema multi-core, isso pode exceder 100% 
                # (ex: 250% se estiver usando 2.5 cores).
                cpu_percent = CURRENT_PROCESS.cpu_percent(interval=self.poll_interval)
                self.max_cpu_percent = max(self.max_cpu_percent, cpu_percent)

                # 2. Uso de Memória (RSS - Resident Set Size)
                # É a memória física real que o processo está usando.
                mem_info = CURRENT_PROCESS.memory_info()
                self.max_mem_rss = max(self.max_mem_rss, mem_info.rss)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # O processo pode ter terminado
                break
            
            # Pequena pausa para evitar que a própria thread de 
            # monitoramento consuma CPU desnecessariamente.
            # (Note: cpu_percent(interval=...) já introduz uma espera)
            # time.sleep(self.poll_interval) # Desnecessário se interval > 0

    def stop(self):
        """Sinaliza para a thread parar."""
        self.stop_event.set()


def save_info_as_csv(info: dict):
    try:
        keys = list(info.keys())
        info_values = list(info.values())

        if not os.path.exists("data/monit_info.csv"):
            with open("data/monit_info.csv", "w", newline="", encoding='utf-8') as f:
                write = csv.writer(f, delimiter=";")
                write.writerow(keys)
                write.writerow(info_values)
        else:
            with open("data/monit_info.csv", "a", newline="", encoding='utf-8') as f:
                write = csv.writer(f, delimiter=";")
                write.writerow(info_values)
    except Exception as ex:
        print(f"{10*"-"} Erro ao salvar informações de monitoramento {10*"-"}")


def execution_monitor(func: Callable) -> Callable:
    """
    Decorator para monitorar tempo, CPU, memória e outras métricas
    essenciais para a análise de paralelismo.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        
        # --- 1. Preparação ---
        monitor = ResourceMonitor(poll_interval=0.05) # Intervalo de 50ms
        
        # Métricas "Antes"
        start_time = time.perf_counter() # Tempo de relógio (wall time)
        start_cpu_times = CURRENT_PROCESS.cpu_times() # Tempo de CPU (user/system)
        start_ctx_switches = CURRENT_PROCESS.num_ctx_switches() # Trocas de Contexto
        
        # Inicia a thread de monitoramento de pico
        monitor.start()

        # --- 2. Execução ---
        try:
            result = func(*args, **kwargs)
        finally:
            # --- 3. Coleta ---
            # Para o monitor o mais rápido possível após a execução
            monitor.stop()
            
            # Métricas "Depois"
            end_time = time.perf_counter()
            end_cpu_times = CURRENT_PROCESS.cpu_times()
            end_ctx_switches = CURRENT_PROCESS.num_ctx_switches()
            
            # Espera a thread do monitor finalizar
            monitor.join()

        # --- 4. Cálculo das Métricas ---
        wall_time = end_time - start_time
        
        # Tempo de CPU total (o tempo que a CPU *realmente* gastou no processo)
        total_cpu_time = (end_cpu_times.user - start_cpu_times.user) + \
                         (end_cpu_times.system - start_cpu_times.system)
        
        # Trocas de contexto
        vol_ctx_switches = end_ctx_switches.voluntary - start_ctx_switches.voluntary
        invol_ctx_switches = end_ctx_switches.involuntary - start_ctx_switches.involuntary
        
        # Taxa de CPU-Bound (Métrica Chave!)
        # Se o tempo de CPU for 0, evitamos divisão por zero.
        if wall_time > 0:
            # Em sistemas multi-core, total_cpu_time pode ser > wall_time
            # Por isso, normalizamos pela contagem de CPUs (lógicas)
            cpu_bound_ratio = (total_cpu_time / wall_time) / psutil.cpu_count()
        else:
            cpu_bound_ratio = 0.0 # Execução instantânea

        # --- 5. Relatório ---
        print("\n" + "="*50)
        print(f"RELATÓRIO DE EXECUÇÃO: {func.__name__}")
        print("="*50)
        
        # Métrica Principal (solicitada)
        print(f"  Tempo de Execução (Wall Time): {wall_time:.4f} s")
        
        # Métricas de Pico (solicitadas)
        print(f"  Pico de Uso de CPU (por 1 core): {monitor.max_cpu_percent:.2f} %")
        print(f"  Pico de Memória (RSS):         {monitor.max_mem_rss / (1024*1024):.2f} MB")
        
        print("-" * 50)
        print("  ANÁLISE PARA PARALELISMO:")
        
        # Métricas Adicionais (o "pulo do gato")
        print(f"  Tempo Total de CPU (User+System): {total_cpu_time:.4f} s")
        print(f"  Taxa de CPU-Bound (Normalizada):  {cpu_bound_ratio:.2%}")
        print(f"  Trocas de Contexto Voluntárias:   {vol_ctx_switches}")
        print(f"  Trocas de Contexto Involuntárias: {invol_ctx_switches}")
        
        print("="*50 + "\n")

        result_dict = {}

        if isinstance(result, dict):
            result_dict = result.copy()

        save_info_as_csv({
            "Função": func.__name__,
            "Tempo de Execução (Wall Time)": f"{wall_time:.4f} s",
            "Pico de Uso de CPU (por 1 core)":  f"{monitor.max_cpu_percent:.2f} %",
            "Pico de Memória (RSS)": f"{monitor.max_mem_rss / (1024*1024):.2f} MB",
            "Tempo Total de CPU (User+System)": f"{total_cpu_time:.4f} s",
            "Taxa de CPU-Bound (Normalizada)": f"{cpu_bound_ratio:.2%}",
            "Trocas de Contexto Voluntárias": f"{vol_ctx_switches}", 
            "Trocas de Contexto Involuntárias": f"{invol_ctx_switches}",
            "Tamanho do arquivo original": result_dict.get("original_file_size"),
            "Tamanho do arquivo .zip": result_dict.get("zip_file_size"),
            "Taxa de compressão": result_dict.get("compression_ratio"),
            "Chunk de leitura (MB)": result_dict.get("chunk_size"),
            "Max Workers (threads)": result_dict.get("max_workers")
        })

        return result
    return wrapper