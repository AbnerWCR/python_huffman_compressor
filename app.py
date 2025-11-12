from modules.huffman_compressor import HuffmanCompressor
from modules.huffman_parallel_compressor import HuffmanParallelCompressor
import os
from modules.resource_monitor import execution_monitor
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

@execution_monitor
def pipeline_compact(arquivo_original, arquivo_zip):

    print(f"Arquivo de amostra '{arquivo_original}' criado com sucesso.")
    print("-" * 40)

    # --- Processo de Compressão ---
    compressor = HuffmanCompressor()
    compressor.compress_file_chunk(arquivo_original, arquivo_zip)

    print("-" * 40)

    tamanho_original = os.path.getsize(arquivo_original)
    tamanho_compactado = os.path.getsize(arquivo_zip)

    if tamanho_original > 0:
        taxa = (1 - tamanho_compactado / tamanho_original) * 100
        # print("Análise Final:")
        # print(f"  Tamanho do arquivo original: {tamanho_original} bytes")
        # print(f"  Tamanho do arquivo .zip:    {tamanho_compactado} bytes")
        # print(f"  Taxa de compressão: {taxa:.2f}%")

        return {
            "original_file_size": f"{tamanho_original} bytes",
            "zip_file_size": f"{tamanho_compactado} bytes",
            "compression_ratio": f"{taxa:.2f}%",
            "chunk_size": 1_000_000 / 1024 / 1024,
            "max_workers": 1,
            "file_name": arquivo_original
        }

@execution_monitor
def pipeline_compact_parallel(arquivo_original, arquivo_zip, chunk_chars, max_workers):

    print(f"Arquivo de amostra '{arquivo_original}' criado com sucesso.")
    print("-" * 40)

    # --- Processo de Compressão ---
    compressor = HuffmanParallelCompressor()
    compressor.compress_file(arquivo_original, arquivo_zip, chunk_chars, max_workers)

    print("-" * 40)

    # --- Verificação e Análise ---
    tamanho_original = os.path.getsize(arquivo_original)
    tamanho_compactado = os.path.getsize(arquivo_zip)

    if tamanho_original > 0:
        taxa = (1 - tamanho_compactado / tamanho_original) * 100
        # print("Análise Final:")
        # print(f"  Tamanho do arquivo original: {tamanho_original} bytes")
        # print(f"  Tamanho do arquivo .zip:    {tamanho_compactado} bytes")
        # print(f"  Taxa de compressão: {taxa:.2f}%")
    
        return {
            "original_file_size": f"{tamanho_original} bytes",
            "zip_file_size": f"{tamanho_compactado} bytes",
            "compression_ratio": f"{taxa:.2f}%",
            "chunk_size": chunk_chars,
            "max_workers": max_workers,
            "file_name": arquivo_original
        }

def pipeline_descompact_parallel(arquivo_original, arquivo_descompactado):

    # --- Processo de Descompressão ---
    # Criamos uma nova instância para simular um programa separado
    decompressor = HuffmanCompressor()
    decompressor.decompress_file(arquivo_zip, arquivo_descompactado)
    print("-" * 40)

    # # Verificar se o conteúdo é idêntico
    with open(arquivo_descompactado, 'r', encoding='utf-8') as f:
        conteudo_descompactado = f.read()
    

    with open(arquivo_original, 'r', encoding='utf-8') as f:
        conteudo_original = f.read()

    if conteudo_original == conteudo_descompactado:
        print("\nVerificação bem-sucedida: O arquivo original e o descompactado são idênticos!")
    else:
        print("\nErro na verificação: Os arquivos são diferentes.")

# @execution_monitor
def execution_single_thread(file_path):
    root_path = Path(file_path)

    files_to_compact = [p for p in root_path.glob("**/*") if p.is_file() and p.name and ".gitkeep" not in p.name]
    files_to_compact = sorted(files_to_compact, key=lambda p: p.name[0].lower())

    if not files_to_compact:
        print("Nenhum arquivo encontrado para compactar.")
        return 0.0
    
    for file in files_to_compact:
        original_path = f"files/{file.stem}.txt"
        zip_path = f"data/compress_files/{file.stem}.zip"

        pipeline_compact(original_path, zip_path)

    
@execution_monitor
def execution_multi_thread(n_thread, file_path):

    root_path = Path(file_path)

    files_to_compact = [p for p in root_path.glob("**/*") if p.is_file()]

    if not files_to_compact:
        print("Nenhum arquivo encontrado para compactar.")
        return 0.0

    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        futures = [
            executor.submit(pipeline_compact, f"files/{file.name}.txt", f"data/{file.name}.zip")
            for file in files_to_compact
        ]

        for f in as_completed(futures):
            print(f"Complete {f.result()}")

# @execution_monitor
def execution_huffman_parallel_compressor(file_path, chunk_chars, max_workers):
    root_path = Path(file_path)

    files_to_compact = [p for p in root_path.glob("**/*") if p.is_file() and ".gitkeep" not in p.name]
    files_to_compact = sorted(files_to_compact, key=lambda p: p.name[0].lower())

    if not files_to_compact:
        print("Nenhum arquivo encontrado para compactar.")
        return 0.0
    
    for file in files_to_compact:
        original_path = f"files/{file.stem}.txt"
        zip_path = f"data/compress_files/{file.stem}-{str(chunk_chars)}-{str(max_workers)}.zip"

        pipeline_compact_parallel(original_path, zip_path, chunk_chars, max_workers)


if __name__ == "__main__":

    arquivo_original = "files/100-livros.txt"
    arquivo_zip = "data/compress_files/100-livros.zip"
    arquivo_descompactado = "data/decompress_files/100-livros_descompactado.txt"

    files_path = "files"

    print("-"*15 + "Execution single" + "-"*15)
    execution_single_thread(files_path)

    # print("-"*15 + "Execution parallel" + "-"*15)
    # print(f"Max Cpu count {max(1, (os.cpu_count() or 1))}")
    # for i in range(2, max(1, (os.cpu_count() or 1))):
    #     # for chunk in range(500, 1_500, 500):
    #     #     print(f"Threads: {i} -- Chunk: {chunk}")
    #     execution_huffman_parallel_compressor(files_path, 1_000_000, i)


    # execution_single_thread(arquivo_original, arquivo_zip)
    # execution_huffman_parallel_compressor(arquivo_original, arquivo_zip)

    # print("-"*50)
    # pipeline_compact(arquivo_original, arquivo_zip)
    # pipeline_compact_parallel(arquivo_original, arquivo_zip, 50, 5)

    # pipeline_descompact_parallel(arquivo_original, arquivo_descompactado)


    
