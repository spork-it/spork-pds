#ifndef SPORK_PDS_INTERNAL_H
#define SPORK_PDS_INTERNAL_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/*
 * Patch 4 direct-access audit (grep: PyList_*, PyTuple_*, PyDict_Next,
 * PyDict_GET_SIZE, and mutable struct fields):
 *
 *   Accessed storage                  Classification / invariant
 *   --------------------------------  ---------------------------------------
 *   Vector tails and iterator pairs  immutable tuples
 *   Function call positional args    immutable tuples
 *   HAMT node lists                   immutable when persistent; editable
 *                                     only for the exact transient token
 *   TransientVector tail lists        owner-confined transient storage
 *   Builder/result lists and tuples   fresh locals, not externally published
 *   Function keyword dictionaries     private call-frame dictionaries
 *   Externally supplied Map pairs     PySequence_GetItem strong references
 *
 * There are no PyDict_Next sites.  Persistent node/value fields are immutable
 * after publication except synchronized caches; transient fields are accessed
 * only after owner validation; iterator cursor fields are synchronized.
 * Consequently direct macros remain only on immutable, private, or owner-
 * confined storage.  Keep this table in sync with new direct accessor sites.
 *
 * HAMT copy-on-write proof: every mutation of an existing node list follows
 * ensure_editable(); fresh unpublished lists are initialized directly.
 * Editability requires a non-NULL token identical to the node token. Persistent
 * operations pass NULL; each transient receives a fresh sentinel. Therefore a
 * published persistent list and two transients from one source can never be
 * writable aliases.
 *
 * Allocation-domain audit:
 *   - Python objects use type allocators or PyObject_New and matching tp_free.
 *   - Typed-vector raw tails/caches use malloc/realloc/free consistently.
 *   - SortedVectorIterator stacks use PyMem_Malloc/Realloc/Free consistently.
 *   - The extension has no direct PyObject_Malloc allocation.
 */

/*
 * Object critical sections were added in CPython 3.13.  They lock the
 * object's per-object mutex in free-threaded builds and are no-ops when the
 * GIL is enabled.  Older supported CPython versions are always GIL-enabled,
 * so use a lexical no-op there as well.
 */
#if PY_VERSION_HEX >= 0x030D0000
#define PDS_BEGIN_CRITICAL_SECTION(op) Py_BEGIN_CRITICAL_SECTION(op)
#define PDS_END_CRITICAL_SECTION() Py_END_CRITICAL_SECTION()
#else
#define PDS_BEGIN_CRITICAL_SECTION(op) { (void)(op)
#define PDS_END_CRITICAL_SECTION() }
#endif

#define PDS_TRANSIENT_WRONG_THREAD_ERROR \
    "Transient objects are confined to their creating thread"

/*
 * Iterator next wrappers hold the iterator's object critical section and use
 * this error when a suspended or reentrant call encounters an active cursor.
 * GC traversal is stop-the-world on free-threaded CPython; clear/deallocation
 * only operate once the iterator is unreachable, so lifecycle hooks do not
 * participate in cursor locking.
 */
#define PDS_ITERATOR_BUSY_ERROR "Iterator is already executing"

/*
 * Transients are single-owner builders.  PyThreadState IDs are stable for a
 * thread state's lifetime and are available throughout our CPython 3.10+
 * support window.  Keep the metadata on every build, but preserve historical
 * regular-GIL behavior by enforcing it only in free-threaded builds.
 */
static inline uint64_t
pds_current_thread_state_id(void)
{
    return PyThreadState_GetID(PyThreadState_Get());
}

static inline int
pds_check_transient_owner(uint64_t owner_thread_id)
{
#ifdef Py_GIL_DISABLED
    if (owner_thread_id != pds_current_thread_state_id()) {
        PyErr_SetString(PyExc_RuntimeError,
                        PDS_TRANSIENT_WRONG_THREAD_ERROR);
        return -1;
    }
#else
    (void)owner_thread_id;
#endif
    return 0;
}

/* A free-threaded CPython object becomes immortal above UINT32_MAX. */
#if PY_VERSION_HEX >= 0x030D0000 && defined(Py_GIL_DISABLED)
#define PDS_SINGLETONS_ARE_IMMORTAL 1
#define PDS_IMMORTAL_REFCNT ((Py_ssize_t)(((size_t)UINT32_MAX) + 1))
#define PDS_SET_IMMORTAL(op) Py_SET_REFCNT((op), PDS_IMMORTAL_REFCNT)
#else
#define PDS_SINGLETONS_ARE_IMMORTAL 0
#define PDS_SET_IMMORTAL(op) ((void)(op))
#endif

#define BITS 5
#define WIDTH (1 << BITS)
#define MASK (WIDTH - 1)

#define ITER_MODE_ITEMS 0
#define ITER_MODE_KEYS 1
#define ITER_MODE_VALUES 2

typedef struct VectorNode {
    PyObject_HEAD
    PyObject *array[WIDTH];
    PyObject *transient_id;
} VectorNode;

typedef struct Vector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    VectorNode *root;
    PyObject *tail;
    Py_hash_t hash;
    int hash_computed;
    PyObject *transient_id;
    int initialized;
} Vector;

typedef struct TransientVector TransientVector;
typedef struct DoubleVectorNode DoubleVectorNode;
typedef struct DoubleVector DoubleVector;
typedef struct TransientDoubleVector TransientDoubleVector;
typedef struct IntVectorNode IntVectorNode;
typedef struct IntVector IntVector;
typedef struct TransientIntVector TransientIntVector;
typedef struct BitmapIndexedNode BitmapIndexedNode;
typedef struct ArrayNode ArrayNode;
typedef struct HashCollisionNode HashCollisionNode;
typedef struct Map Map;
typedef struct TransientMap TransientMap;
typedef struct Set Set;
typedef struct TransientSet TransientSet;
typedef struct SetIterator SetIterator;

typedef struct Cons {
    PyObject_HEAD
    PyObject *first;
    PyObject *rest;
    Py_hash_t hash;
    int hash_computed;
    int initialized;
} Cons;

typedef struct RBNode RBNode;
typedef struct SortedVector {
    PyObject_HEAD
    RBNode *root;
    Py_ssize_t cnt;
    PyObject *key_fn;
    int reverse;
} SortedVector;
typedef struct TransientSortedVector TransientSortedVector;
typedef struct SortedVectorIterator SortedVectorIterator;

/* Shared utility type and helpers. */
extern PyTypeObject PdsSentinelType;
extern PyObject *_MISSING;

PyObject *Generic_class_getitem(PyObject *cls, PyObject *args);
PyObject *pds_empty_iterator(void);

static inline Py_hash_t
pds_finalize_hash(Py_uhash_t hash)
{
    Py_hash_t result = (Py_hash_t)hash;
    return result == -1 ? -2 : result;
}

int ctpop(unsigned int i);
int mask_hash(Py_hash_t hash_val, int shift);
unsigned int bitpos(Py_hash_t hash_val, int shift);
int bitmap_index(unsigned int bitmap, unsigned int bit);

/* Type objects used during module initialization and across implementations. */
extern PyTypeObject ConsType;
extern PyTypeObject ConsIteratorType;
extern PyTypeObject VectorNodeType;
extern PyTypeObject VectorType;
extern PyTypeObject VectorIteratorType;
extern PyTypeObject TransientVectorType;
extern PyTypeObject TransientVectorIteratorType;
extern PyTypeObject DoubleVectorNodeType;
extern PyTypeObject DoubleVectorType;
extern PyTypeObject DoubleVectorIteratorType;
extern PyTypeObject TransientDoubleVectorType;
extern PyTypeObject IntVectorNodeType;
extern PyTypeObject IntVectorType;
extern PyTypeObject IntVectorIteratorType;
extern PyTypeObject TransientIntVectorType;
extern PyTypeObject BitmapIndexedNodeType;
extern PyTypeObject ArrayNodeType;
extern PyTypeObject HashCollisionNodeType;
extern PyTypeObject BitmapIndexedNodeIteratorType;
extern PyTypeObject ArrayNodeIteratorType;
extern PyTypeObject HashCollisionNodeIteratorType;
extern PyTypeObject MapType;
extern PyTypeObject TransientMapType;
extern PyTypeObject SetType;
extern PyTypeObject TransientSetType;
extern PyTypeObject SetIteratorType;
extern PyTypeObject RBNodeType;
extern PyTypeObject SortedVectorType;
extern PyTypeObject SortedVectorIteratorType;
extern PyTypeObject TransientSortedVectorType;

/* Process-wide singleton aliases populated by module initialization. */
extern VectorNode *EMPTY_NODE;
extern Vector *EMPTY_VECTOR;
extern DoubleVectorNode *EMPTY_DOUBLE_NODE;
extern DoubleVector *EMPTY_DOUBLE_VECTOR;
extern IntVectorNode *EMPTY_LONG_NODE;
extern IntVector *EMPTY_LONG_VECTOR;
extern BitmapIndexedNode *EMPTY_BIN;
extern Map *EMPTY_MAP;
extern Set *EMPTY_SET;
extern SortedVector *EMPTY_SORTED_VECTOR;

/* Constructors needed by module initialization. */
VectorNode *VectorNode_create(PyObject *transient_id);
Vector *Vector_create(Py_ssize_t cnt, int shift, VectorNode *root,
                      PyObject *tail, PyObject *transient_id);
DoubleVectorNode *DoubleVectorNode_create(PyObject *transient_id);
DoubleVector *DoubleVector_create(Py_ssize_t cnt, int shift,
                                  DoubleVectorNode *root, double *tail,
                                  Py_ssize_t tail_len, PyObject *transient_id);
IntVectorNode *IntVectorNode_create(PyObject *transient_id);
IntVector *IntVector_create(Py_ssize_t cnt, int shift, IntVectorNode *root,
                            int64_t *tail, Py_ssize_t tail_len,
                            PyObject *transient_id);
BitmapIndexedNode *BitmapIndexedNode_create(unsigned int bitmap,
                                            PyObject *array,
                                            PyObject *transient_id);
Map *Map_create(Py_ssize_t cnt, PyObject *root, PyObject *transient_id);
Set *Set_create(Py_ssize_t cnt, PyObject *root, PyObject *transient_id);

/* Vector operations used by Map's optimized pair handling. */
Py_ssize_t Vector_length(Vector *self);
PyObject *Vector_nth_impl(Vector *self, Py_ssize_t i, PyObject *default_val);

/* HAMT operations shared by Map and Set. */
BitmapIndexedNode *BitmapIndexedNode_ensure_editable(
    BitmapIndexedNode *self, PyObject *transient_id);
ArrayNode *ArrayNode_ensure_editable(ArrayNode *self, PyObject *transient_id);
HashCollisionNode *HashCollisionNode_ensure_editable(
    HashCollisionNode *self, PyObject *transient_id);
PyObject *BitmapIndexedNode_assoc(BitmapIndexedNode *self, int shift,
                                  Py_hash_t hash_val, PyObject *key,
                                  PyObject *val, PyObject *added_leaf,
                                  PyObject *transient_id);
PyObject *BitmapIndexedNode_find(BitmapIndexedNode *self, int shift,
                                 Py_hash_t hash_val, PyObject *key,
                                 PyObject *not_found);
PyObject *BitmapIndexedNode_dissoc(BitmapIndexedNode *self, int shift,
                                   Py_hash_t hash_val, PyObject *key,
                                   PyObject *removed_leaf,
                                   PyObject *transient_id);
PyObject *ArrayNode_assoc(ArrayNode *self, int shift, Py_hash_t hash_val,
                          PyObject *key, PyObject *val,
                          PyObject *added_leaf, PyObject *transient_id);
PyObject *ArrayNode_find(ArrayNode *self, int shift, Py_hash_t hash_val,
                         PyObject *key, PyObject *not_found);
PyObject *ArrayNode_dissoc(ArrayNode *self, int shift, Py_hash_t hash_val,
                           PyObject *key, PyObject *removed_leaf,
                           PyObject *transient_id);
PyObject *HashCollisionNode_assoc(HashCollisionNode *self, int shift,
                                  Py_hash_t hash_val, PyObject *key,
                                  PyObject *val, PyObject *added_leaf,
                                  PyObject *transient_id);
PyObject *HashCollisionNode_find(HashCollisionNode *self, int shift,
                                 Py_hash_t hash_val, PyObject *key,
                                 PyObject *not_found);
PyObject *HashCollisionNode_dissoc(HashCollisionNode *self, int shift,
                                   Py_hash_t hash_val, PyObject *key,
                                   PyObject *removed_leaf,
                                   PyObject *transient_id);
PyObject *BitmapIndexedNode_iter_mode(BitmapIndexedNode *self, int mode);
PyObject *ArrayNode_iter_mode(ArrayNode *self, int mode);
PyObject *HashCollisionNode_iter_mode(HashCollisionNode *self, int mode);
PyObject *BitmapIndexedNode_iter_mode_transient(
    BitmapIndexedNode *self, int mode, uint64_t owner_thread_id);
PyObject *ArrayNode_iter_mode_transient(
    ArrayNode *self, int mode, uint64_t owner_thread_id);
PyObject *HashCollisionNode_iter_mode_transient(
    HashCollisionNode *self, int mode, uint64_t owner_thread_id);
PyObject *BitmapIndexedNode_iter_kv(BitmapIndexedNode *self);
PyObject *ArrayNode_iter_kv(ArrayNode *self);
PyObject *HashCollisionNode_iter_kv(HashCollisionNode *self);

/* Python-level factories installed in the module method table. */
PyObject *pds_cons(PyObject *self, PyObject *args);
PyObject *pds_vec(PyObject *self, PyObject *args);
PyObject *pds_vec_f64(PyObject *self, PyObject *args);
PyObject *pds_vec_i64(PyObject *self, PyObject *args);
PyObject *pds_hash_map(PyObject *self, PyObject *args);
PyObject *pds_set(PyObject *self, PyObject *args);
PyObject *pds_sorted_vec(PyObject *self, PyObject *args, PyObject *kwargs);

#endif /* SPORK_PDS_INTERNAL_H */
