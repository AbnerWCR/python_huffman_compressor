import heapq
from collections import Counter
import json
import zipfile
import os
import psutil
from models.huffman_node import HuffmanNode
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
import gc

def _count_chunk_text(chunk: str) -> Dict[str, int]:
    """Worker for parallel counting (top-level for pickling)."""
    return dict(Counter(chunk))

class HuffmanParallelCompressor:
    """
    Parallel-aware Huffman compressor:
    - Parallelizes frequency counting by partitioning the input text.
    - Streams encoding to avoid holding the whole encoded bitstring in memory.
    - Produces the same archive format: header.json (reverse mapping) + data.bin
      where data.bin starts with one padding byte followed by encoded bytes.
    """

    def __init__(self, memory_threshold_percent: float = 70.0):
        self.codes = {}
        self.reverse_mapping = {}
        self.memory_threshold_percent = memory_threshold_percent

    def _generate_codes_recursive(self, node, current_code):
        if node is None:
            return
        if node.char is not None:
            self.codes[node.char] = current_code
            self.reverse_mapping[current_code] = node.char
            return
        self._generate_codes_recursive(node.left, current_code + "0")
        self._generate_codes_recursive(node.right, current_code + "1")

    def _generate_codes_iterative(self, root):
        tree_stack = [(root, "")]

        while tree_stack:
            node, code = tree_stack.pop()

            if node is None:
                continue

            if node.char is not None:
                self.codes[node.char] = code
                self.reverse_mapping[code] = node.char
            else:
                tree_stack.append((node.left, code+"0",))
                tree_stack.append((node.right, code+"1",))

    # --- Bit helpers ---
    def _get_bytearray_from_bits(self, bits: str) -> bytearray:
        """Convert a bit string whose length is a multiple of 8 into a bytearray."""
        b = bytearray()
        for i in range(0, len(bits), 8):
            b.append(int(bits[i:i+8], 2))
        return b

    def _get_available_memory_percent(self) -> float:
        """Get percentage of available memory."""
        return psutil.virtual_memory().percent

    def _get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)

    def _should_process_chunk(self) -> bool:
        """Check if we have enough memory to process the next chunk."""
        current_percent = self._get_available_memory_percent()
        available_mb = self._get_available_memory_mb()
        
        if current_percent > self.memory_threshold_percent:
            return True
        
        print(f"⚠️  Memória disponível: {available_mb:.2f} MB ({100 - current_percent:.2f}% livre)")
        return False

    # --- Main functions ---
    def compress_file(self, input_file_path: str, output_file_path: str, chunk_chars: int = 1_000_000, max_workers=None):
        """
        Compress input_file_path to output_file_path (.zip).
        chunk_chars: initial number of characters to read per partition.
        Dynamically adjusted based on available memory.
        """
        print(f"Iniciando compressão de '{input_file_path}'...")
        print(f"Memória disponível: {self._get_available_memory_mb():.2f} MB ({100 - self._get_available_memory_percent():.2f}% livre)")

        if not os.path.exists(input_file_path):
            print(f"Erro: Arquivo '{input_file_path}' não encontrado.")
            return

        # 1) Parallel frequency counting
        total_counter = Counter()

        if max_workers is None:
            max_workers = max(1, (os.cpu_count() or 1) - 1)

        futures = []
        current_chunk_size = chunk_chars
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                while True:
                    # Check memory before reading chunk
                    if not self._should_process_chunk():
                        # Reduce chunk size if memory is low
                        current_chunk_size = max(100_000, current_chunk_size // 2)
                        print(f"📉 Reduzindo tamanho do chunk para: {current_chunk_size:,} caracteres")
                    
                    chunk = f.read(current_chunk_size)
                    if not chunk:
                        break
                    futures.append(executor.submit(_count_chunk_text, chunk))

            for fut in as_completed(futures):
                part = fut.result()
                total_counter.update(part)

        if not total_counter:
            # empty file -> create empty archive
            with zipfile.ZipFile(output_file_path, 'w') as zipf:
                zipf.writestr('header.json', json.dumps({}))
                zipf.writestr('data.bin', bytes([0]))
            print(f"Compressão concluída! Arquivo salvo em '{output_file_path}'")
            return output_file_path

        # 2) Build Huffman tree and codes (single-threaded; small overhead)
        print("Construindo árvore de Huffman...")
        priority_queue = [HuffmanNode(char, freq) for char, freq in total_counter.items()]
        heapq.heapify(priority_queue)

        while len(priority_queue) > 1:
            left = heapq.heappop(priority_queue)
            right = heapq.heappop(priority_queue)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left, merged.right = left, right
            heapq.heappush(priority_queue, merged)

        root = priority_queue[0]
        self.codes.clear()
        self.reverse_mapping.clear()
        self._generate_codes_iterative(root)
        print(f"Árvore criada com {len(self.codes)} caracteres únicos")

        # 3) Stream-encode the file (no global bitstring)
        print("Codificando arquivo...")
        bit_buffer = ""
        out_bytes = bytearray()
        flush_bits_threshold = 4 * 1024 * 1024  # Flush with 4MB (reduzido de 8MB)
        current_chunk_size = chunk_chars
        bytes_written = 0

        with open(input_file_path, 'r', encoding='utf-8') as f:
            while True:
                # Check memory before reading chunk
                if not self._should_process_chunk():
                    current_chunk_size = max(100_000, current_chunk_size // 2)
                    print(f"📉 Reduzindo tamanho do chunk para: {current_chunk_size:,} caracteres")
                
                chunk = f.read(current_chunk_size)
                if not chunk:
                    break
                
                # Map characters to codes
                parts = [self.codes[ch] for ch in chunk]
                bit_buffer += "".join(parts)

                # Flush full bytes from buffer to out_bytes when threshold reached
                if len(bit_buffer) >= flush_bits_threshold:
                    n = (len(bit_buffer) // 8) * 8
                    out_bytes.extend(self._get_bytearray_from_bits(bit_buffer[:n]))
                    bit_buffer = bit_buffer[n:]
                    bytes_written += len(out_bytes)
                    print(f"💾 Codificados {bytes_written:,} bytes | Buffer: {len(bit_buffer)} bits | Memória: {self._get_available_memory_mb():.2f} MB")

        # 4) Finalize padding
        print("Finalizando codificação...")
        extra_padding = (8 - (len(bit_buffer) % 8)) % 8
        if extra_padding:
            bit_buffer += '0' * extra_padding
        if bit_buffer:
            out_bytes.extend(self._get_bytearray_from_bits(bit_buffer))
        
        del bit_buffer
        gc.collect()

        # Prepend single padding info byte
        final_data = bytes([extra_padding]) + bytes(out_bytes)
        del extra_padding
        del out_bytes
        gc.collect()

        # 5) Write ZIP with header and data
        print("Salvando arquivo comprimido...")
        header = json.dumps(self.reverse_mapping)
        with zipfile.ZipFile(output_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('header.json', header)
            zipf.writestr('data.bin', final_data)

        print(f"✅ Compressão concluída! Arquivo salvo em '{output_file_path}'")
        print(f"Tamanho final: {os.path.getsize(output_file_path) / (1024*1024):.2f} MB")
        return output_file_path

    def decompress_file(self, input_file_path: str, output_file_path: str):
        """Descomprime o .zip gerado pelo compressor."""
        print(f"Iniciando descompressão de '{input_file_path}'...")

        with zipfile.ZipFile(input_file_path, 'r') as zipf:
            header_json = zipf.read('header.json').decode('utf-8')
            self.reverse_mapping = json.loads(header_json)

            byte_array = zipf.read('data.bin')
            if not byte_array:
                # nothing to do
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write("")
                print(f"Descompressão concluída! Arquivo salvo em '{output_file_path}'")
                return output_file_path

            # first byte is padding info
            extra_padding = byte_array[0]
            data_bytes = byte_array[1:]

            bits_string = "".join(f"{byte:08b}" for byte in data_bytes)
            if extra_padding:
                bits_string = bits_string[:-extra_padding]

            # decode bits
            decoded_chars = []
            current_code = ""
            for bit in bits_string:
                current_code += bit
                if current_code in self.reverse_mapping:
                    decoded_chars.append(self.reverse_mapping[current_code])
                    current_code = ""

            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write("".join(decoded_chars))

        print(f"Descompressão concluída! Arquivo salvo em '{output_file_path}'")
        return output_file_path