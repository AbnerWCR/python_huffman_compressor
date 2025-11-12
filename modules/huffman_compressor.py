import heapq
from collections import Counter
import json
import zipfile
import os
from models.huffman_node import HuffmanNode
import gc
from typing import Dict
import psutil

class HuffmanCompressor:
    def __init__(self, memory_threshold_percent: float = 70.0):
        self.codes = {}
        self.reverse_mapping = {}
        self.memory_threshold_percent = memory_threshold_percent

    def _generate_codes_recursive(self, node, current_code):
        if node is None: return
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

    # --- Funções internas para manipulação de bits e bytes ---
    def _pad_encoded_text(self, encoded_text):
        """ Adiciona preenchimento para que o texto codificado seja múltiplo de 8. """
        extra_padding = 8 - len(encoded_text) % 8
        if extra_padding == 8: extra_padding = 0

        padded_encoded_text = encoded_text + ('0' * extra_padding)
        padding_info = "{0:08b}".format(extra_padding) # Salva a informação de padding como um byte

        return padded_encoded_text, padding_info

    def _get_byte_array(self, padded_encoded_text):
        """ Converte a string de bits preenchida em um array de bytes. """
        if len(padded_encoded_text) % 8 != 0:
            print("Erro: Texto codificado não foi preenchido corretamente.")
            exit()

        b = bytearray()
        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i+8]
            b.append(int(byte, 2))
        return b

    def _remove_padding(self, padded_encoded_text):
        """ Remove o preenchimento do texto decodificado. """
        padding_info = padded_encoded_text[:8]
        extra_padding = int(padding_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-extra_padding] if extra_padding != 0 else padded_encoded_text
        return encoded_text
    
    def _count_chunk_text(self, chunk: str) -> Dict[str, int]:
        """Worker for parallel counting (top-level for pickling)."""
        return dict(Counter(chunk))
    
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
    
    def _get_bytearray_from_bits(self, bits: str) -> bytearray:
        """Convert a bit string whose length is a multiple of 8 into a bytearray."""
        b = bytearray()
        for i in range(0, len(bits), 8):
            b.append(int(bits[i:i+8], 2))
        return b
    
    def compress_file_chunk(self, input_file_path, output_file_path, chunk_chars=1_000_000):
        """ Comprime um arquivo de texto e salva o resultado em um .zip. """
        print(f"Iniciando compressão de '{input_file_path}'...")

        # 1. Ler o texto do arquivo
        frequency = Counter()
        # try:
        #     with open(input_file_path, 'r', encoding='utf-8') as f:
        #         text = f.read()
        # except FileNotFoundError:
        #     print(f"Erro: Arquivo '{input_file_path}' não encontrado.")
        #     return

        current_chunk_size = chunk_chars
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            try:
                while True:
                    if not self._should_process_chunk():
                        # Reduce chunk size if memory is low
                        current_chunk_size = max(100_000, current_chunk_size // 2)
                        print(f"📉 Reduzindo tamanho do chunk para: {current_chunk_size:,} caracteres")
                    
                    chunk = f.read(current_chunk_size)
                    if not chunk:
                        break
                    part = self._count_chunk_text(chunk)
                    frequency.update(part)
            except FileNotFoundError:
                print(f"Erro: Arquivo '{input_file_path}' não encontrado.")
                return

        # 2. Construir a árvore e gerar os códigos (lógica original)
        priority_queue = [HuffmanNode(char, freq) for char, freq in frequency.items()]
        heapq.heapify(priority_queue)

        while len(priority_queue) > 1:
            left = heapq.heappop(priority_queue)
            right = heapq.heappop(priority_queue)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left, merged.right = left, right
            heapq.heappush(priority_queue, merged)

        root = priority_queue[0]
        # self._generate_codes_recursive(root, "")
        self._generate_codes_iterative(root)

        # 3) Stream-encode the file (no global bitstring)
        bit_buffer = ""  # keep less than one chunk worth of bits
        out_bytes = bytearray()
        flush_bits_threshold = 8 * 1024 * 1024  # flush when we have >= 8MB of bits

        with open(input_file_path, 'r', encoding='utf-8') as f:
            while True:
                if not self._should_process_chunk():
                    # Reduce chunk size if memory is low
                    current_chunk_size = max(100_000, current_chunk_size // 2)
                    print(f"📉 Reduzindo tamanho do chunk para: {current_chunk_size:,} caracteres")
                
                chunk = f.read(current_chunk_size)
                if not chunk:
                    break
                # Map characters to codes
                parts = [self.codes[ch] for ch in chunk]
                bit_buffer += "".join(parts)

                # Flush full bytes from buffer to out_bytes
                if len(bit_buffer) >= flush_bits_threshold:
                    n = (len(bit_buffer) // 8) * 8
                    out_bytes.extend(self._get_bytearray_from_bits(bit_buffer[:n]))
                    bit_buffer = bit_buffer[n:]

        # 4) Finalize padding
        extra_padding = (8 - (len(bit_buffer) % 8)) % 8
        if extra_padding:
            bit_buffer += '0' * extra_padding
        if bit_buffer:
            out_bytes.extend(self._get_bytearray_from_bits(bit_buffer))

        # Prepend single padding info byte
        final_data = bytes([extra_padding]) + bytes(out_bytes)

        # 5) Write ZIP with header and data
        header = json.dumps(self.reverse_mapping)
        with zipfile.ZipFile(output_file_path, 'w') as zipf:
            zipf.writestr('header.json', header)
            zipf.writestr('data.bin', final_data)

        print(f"Compressão concluída! Arquivo salvo em '{output_file_path}'")
        return output_file_path

    # --- Funções principais de compressão e descompressão de ARQUIVOS ---
    def compress_file(self, input_file_path, output_file_path, chunk_chars=1_000_000):
        """ Comprime um arquivo de texto e salva o resultado em um .zip. """
        print(f"Iniciando compressão de '{input_file_path}'...")

        # 1. Ler o texto do arquivo
        frequency = Counter()
        # try:
        #     with open(input_file_path, 'r', encoding='utf-8') as f:
        #         text = f.read()
        # except FileNotFoundError:
        #     print(f"Erro: Arquivo '{input_file_path}' não encontrado.")
        #     return
        
        text = ""
        with open(input_file_path, 'r', encoding='utf-8') as f:
            try:
                while True:
                    chunk = f.read(1_000_000)
                    if not chunk:
                        break
                    part = self._count_chunk_text(chunk)
                    frequency.update(part)
                    text = text+part
            except FileNotFoundError:
                print(f"Erro: Arquivo '{input_file_path}' não encontrado.")
                return

        # 2. Construir a árvore e gerar os códigos (lógica original)
        priority_queue = [HuffmanNode(char, freq) for char, freq in frequency.items()]
        heapq.heapify(priority_queue)

        while len(priority_queue) > 1:
            left = heapq.heappop(priority_queue)
            right = heapq.heappop(priority_queue)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left, merged.right = left, right
            heapq.heappush(priority_queue, merged)

        root = priority_queue[0]
        # self._generate_codes_recursive(root, "")
        self._generate_codes_iterative(root)

        # 3. Codificar o texto para uma string de bits
        encoded_text = "".join([self.codes[char] for char in text])
        del text
        gc.collect()

        # 4. Adicionar padding e converter para bytes
        padded_text, padding_info = self._pad_encoded_text(encoded_text)
        byte_array = self._get_byte_array(padding_info + padded_text)

        del padded_text
        del padding_info
        gc.collect()

        # 5. Criar o arquivo ZIP e salvar os dados
        with zipfile.ZipFile(output_file_path, 'w') as zipf:
            # O "cabeçalho" com o mapa para descompressão
            # Usamos JSON para serializar o dicionário de forma legível
            header = json.dumps(self.reverse_mapping)
            zipf.writestr('header.json', header)

            # Os dados compactados
            zipf.writestr('data.bin', bytes(byte_array))

        print(f"Compressão concluída! Arquivo salvo em '{output_file_path}'")
        return output_file_path

    def decompress_file(self, input_file_path, output_file_path):
        """ Descomprime um arquivo .zip gerado pelo compressor. """
        print(f"Iniciando descompressão de '{input_file_path}'...")

        with zipfile.ZipFile(input_file_path, 'r') as zipf:
            # 1. Ler o cabeçalho e os dados de dentro do zip
            header_json = zipf.read('header.json').decode('utf-8')
            self.reverse_mapping = json.loads(header_json)

            byte_array = zipf.read('data.bin')

            # 2. Converter bytes de volta para a string de bits
            bits_string = "".join([f"{byte:08b}" for byte in byte_array])

            # 3. Remover o padding
            encoded_text = self._remove_padding(bits_string)

            # 4. Decodificar a string de bits
            decoded_text = ""
            current_code = ""
            for bit in encoded_text:
                current_code += bit
                if current_code in self.reverse_mapping:
                    character = self.reverse_mapping[current_code]
                    decoded_text += character
                    current_code = ""

            # 5. Salvar o texto descompactado
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(decoded_text)

        print(f"Descompressão concluída! Arquivo salvo em '{output_file_path}'")
        return output_file_path