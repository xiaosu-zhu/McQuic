import cython

cdef extern from "rans64.h":
    ctypedef uint64_t Rans64State;
    static inline void Rans64EncInit(Rans64State* r)
    static inline void Rans64EncPut(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
    static inline void Rans64EncFlush(Rans64State* r, uint32_t** pptr)
    static inline void Rans64DecInit(Rans64State* r, uint32_t** pptr)
    static inline uint32_t Rans64DecGet(Rans64State* r, uint32_t scale_bits)
    static inline void Rans64DecAdvance(Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits)
