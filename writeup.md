# 2
## 2.1
### a
chr(0) is the unicode null character.

### b
The string representation shows the escape character "\x00", while the printed version shows the raw character which is not visible

### c
In python, when we print it it's not visible, but in other languages and libraries, it may be used as a string terminator. 

## 2.2
### a
UTF-8 uses between 1-4 bytes, UTF-16 uses either 2 or 4 bytes, and UTF-32 uses a fixed 4 bytes for each character. UTF-8 is ASCII compatible and is more compact and reduces the waste of other, which is needed to reduce the sequence length for a model

### b
This function decodes bytes one at a time, meaning we can only decode byte 0 to 255. In reality, actual characters are represented by multiple bytes in sequence and we need to capture those by decoding the entire sequence. An example would be 你好.

### c
'\xff\xff' because \xff is not a valid leading byte in utf-8 since it has a specific bit pattern to tell the decoder how many bytes follow.
