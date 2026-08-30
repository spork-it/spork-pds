#ifndef SPORK_PDS_INTERNAL_H
#define SPORK_PDS_INTERNAL_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#define HAVE_STDATOMIC 1
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

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
