// pds.c - Persistent Data Structures for spork-pds
// C implementation of Vector (bit-partitioned trie) and Map (HAMT)
// Includes type-specialized vectors: DoubleVector, IntVector
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdint.h>

// For atomic operations in free-threaded Python 3.13+
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#define HAVE_STDATOMIC 1
#endif

// Guard to ensure singletons are only created once (for sub-interpreter safety)
static int _singletons_initialized = 0;

// =============================================================================
// IMMORTALIZATION HELPER
// =============================================================================
// Per Python 3.14 docs: "On Python build with Free Threading, if refcnt is
// larger than UINT32_MAX, the object is made immortal."

#if PY_VERSION_HEX >= 0x030D0000 && defined(Py_GIL_DISABLED)
#define PDS_SINGLETONS_ARE_IMMORTAL 1
#define PDS_IMMORTAL_REFCNT ((Py_ssize_t)(((size_t)UINT32_MAX) + 1))
#define PDS_SET_IMMORTAL(op) Py_SET_REFCNT((op), PDS_IMMORTAL_REFCNT)
#else
#define PDS_SINGLETONS_ARE_IMMORTAL 0
#define PDS_SET_IMMORTAL(op) ((void)(op))
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

// =============================================================================
// SENTINEL TYPE
// =============================================================================
// A minimal type for identity/sentinel objects.

typedef struct {
    PyObject_HEAD
} PdsSentinel;

static void PdsSentinel_dealloc(PdsSentinel *self) {
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject PdsSentinelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds._Sentinel",
    .tp_basicsize = sizeof(PdsSentinel),
    .tp_dealloc = (destructor)PdsSentinel_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// === Generic __class_getitem__ for type annotations ===
// This enables Vector[int], Map[str, int], Cons[T] syntax
static PyObject *
Generic_class_getitem(PyObject *cls, PyObject *args)
{
    // Use types.GenericAlias to create a parameterized type
    // This is the standard way to support generics in Python 3.9+
    PyObject *typing_module = PyImport_ImportModule("typing");
    if (typing_module == NULL) {
        return NULL;
    }

    PyObject *generic_alias_func = PyObject_GetAttrString(typing_module, "_GenericAlias");
    Py_DECREF(typing_module);

    if (generic_alias_func == NULL) {
        // Fallback for older Python: just return a tuple (cls, args)
        PyErr_Clear();
        return Py_BuildValue("(OO)", cls, args);
    }

    // Ensure args is a tuple
    PyObject *args_tuple;
    if (PyTuple_Check(args)) {
        args_tuple = args;
        Py_INCREF(args_tuple);
    } else {
        args_tuple = PyTuple_Pack(1, args);
        if (args_tuple == NULL) {
            Py_DECREF(generic_alias_func);
            return NULL;
        }
    }

    // Call _GenericAlias(cls, args_tuple)
    PyObject *result = PyObject_CallFunctionObjArgs(generic_alias_func, cls, args_tuple, NULL);

    Py_DECREF(generic_alias_func);
    Py_DECREF(args_tuple);

    return result;
}

// === Constants for 32-way branching tries ===
#define BITS 5
#define WIDTH (1 << BITS)  // 32
#define MASK (WIDTH - 1)   // 0x1f

// =============================================================================
// TYPE-SPECIALIZED VECTOR MACRO SYSTEM
// =============================================================================
//
// This macro system generates Vector, DoubleVector, and IntVector
// from a single template, avoiding code duplication while supporting:
// - PyObject* storage (standard vectors)
// - double storage (for NumPy float64 interop)
// - long storage (for NumPy int64 interop)
//
// The key differences between object and primitive vectors:
// 1. Reference counting: objects need INCREF/DECREF, primitives don't
// 2. Boxing/unboxing: primitives must be converted to/from PyObject* at API boundary
// 3. Buffer protocol: primitives can expose contiguous memory for zero-copy NumPy access

// === Reference Counting Macros ===
// For PyObject* vectors, these map to Python's ref counting
// For primitive vectors, these are no-ops

#define ITEM_INCREF_OBJ(x)    Py_INCREF(x)
#define ITEM_DECREF_OBJ(x)    Py_DECREF(x)
#define ITEM_XINCREF_OBJ(x)   Py_XINCREF(x)
#define ITEM_XDECREF_OBJ(x)   Py_XDECREF(x)

#define ITEM_INCREF_PRIM(x)   ((void)0)
#define ITEM_DECREF_PRIM(x)   ((void)0)
#define ITEM_XINCREF_PRIM(x)  ((void)0)
#define ITEM_XDECREF_PRIM(x)  ((void)0)

// === Token Pasting Helpers ===
#define CONCAT2(a, b) a ## b
#define CONCAT3(a, b, c) a ## b ## c
#define MAKE_NAME(prefix, name) CONCAT2(prefix, name)
#define MAKE_NAME3(a, b, c) CONCAT3(a, b, c)

// === Utility functions ===
static inline int ctpop(unsigned int i) {
#ifdef __GNUC__
    return __builtin_popcount(i);
#elif defined(_MSC_VER)
    return __popcnt(i);
#else
    int count = 0;
    while (i) {
        count += i & 1;
        i >>= 1;
    }
    return count;
#endif
}

static inline int mask_hash(Py_hash_t hash_val, int shift) {
    return (hash_val >> shift) & MASK;
}

static inline unsigned int bitpos(Py_hash_t hash_val, int shift) {
    return 1U << mask_hash(hash_val, shift);
}

static inline int bitmap_index(unsigned int bitmap, unsigned int bit) {
    return ctpop(bitmap & (bit - 1));
}

// === Forward declarations ===
typedef struct VectorNode VectorNode;
typedef struct Vector Vector;
typedef struct TransientVector TransientVector;
typedef struct DoubleVectorNode DoubleVectorNode;
typedef struct DoubleVector DoubleVector;
typedef struct TransientDoubleVector TransientDoubleVector;
typedef struct IntVectorNode IntVectorNode;
typedef struct IntVector IntVector;
typedef struct TransientIntVector TransientIntVector;
typedef struct MapNode MapNode;
typedef struct BitmapIndexedNode BitmapIndexedNode;
typedef struct ArrayNode ArrayNode;
typedef struct HashCollisionNode HashCollisionNode;
typedef struct Map Map;
typedef struct TransientMap TransientMap;
typedef struct Set Set;
typedef struct TransientSet TransientSet;
typedef struct SetIterator SetIterator;
typedef struct Cons Cons;
typedef struct RBNode RBNode;
typedef struct SortedVector SortedVector;
typedef struct TransientSortedVector TransientSortedVector;
typedef struct SortedVectorIterator SortedVectorIterator;

// Sentinel for missing values
static PyObject *_MISSING = NULL;

// =============================================================================
// MODULE STATE FOR MULTI-PHASE INITIALIZATION
// =============================================================================
//
// Per-module state struct holds singletons that were previously globals.
// This enables proper GC integration and future subinterpreter support.

typedef struct {
    PyObject *_MISSING;

    PyObject *EMPTY_VECTOR;
    PyObject *EMPTY_DOUBLE_VECTOR;
    PyObject *EMPTY_LONG_VECTOR;

    PyObject *EMPTY_MAP;
    PyObject *EMPTY_SET;

    PyObject *EMPTY_SORTED_VECTOR;

    // Internal nodes (not exposed to Python, but needed for cleanup)
    PyObject *EMPTY_NODE;
    PyObject *EMPTY_DOUBLE_NODE;
    PyObject *EMPTY_LONG_NODE;
    PyObject *EMPTY_BIN;
} PdsState;

static inline PdsState *
pds_get_state(PyObject *module)
{
    return (PdsState *)PyModule_GetState(module);
}

// === Cons ===
typedef struct Cons {
    PyObject_HEAD
    PyObject *first;
    PyObject *rest;
    Py_hash_t hash;
    int hash_computed;
} Cons;

static PyTypeObject ConsType;

static void Cons_dealloc(Cons *self) {
    Py_XDECREF(self->first);
    Py_XDECREF(self->rest);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Cons_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Cons *self = (Cons *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->first = Py_None;
        Py_INCREF(Py_None);
        self->rest = Py_None;
        Py_INCREF(Py_None);
        self->hash = 0;
        self->hash_computed = 0;
    }
    return (PyObject *)self;
}

static int Cons_init(Cons *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"first", "rest", NULL};
    PyObject *first = NULL, *rest = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &first, &rest)) {
        return -1;
    }

    Py_XDECREF(self->first);
    self->first = first;
    Py_INCREF(first);

    Py_XDECREF(self->rest);
    if (rest != NULL) {
        self->rest = rest;
        Py_INCREF(rest);
    } else {
        self->rest = Py_None;
        Py_INCREF(Py_None);
    }

    self->hash_computed = 0;
    return 0;
}

static PyObject *Cons_get_first(Cons *self, void *closure) {
    Py_INCREF(self->first);
    return self->first;
}

static PyObject *Cons_get_rest(Cons *self, void *closure) {
    Py_INCREF(self->rest);
    return self->rest;
}

static PyGetSetDef Cons_getsetters[] = {
    {"first", (getter)Cons_get_first, NULL, "First element", NULL},
    {"rest", (getter)Cons_get_rest, NULL, "Rest of the list", NULL},
    {"_first", (getter)Cons_get_first, NULL, "First element (internal)", NULL},
    {"_rest", (getter)Cons_get_rest, NULL, "Rest of the list (internal)", NULL},
    {NULL}
};

static PyObject *Cons_iter(Cons *self);

static Py_ssize_t Cons_length(Cons *self) {
    Py_ssize_t count = 0;
    PyObject *curr = (PyObject *)self;
    while (curr != Py_None && Py_TYPE(curr) == &ConsType) {
        count++;
        curr = ((Cons *)curr)->rest;
    }
    return count;
}

static Py_hash_t Cons_hash(Cons *self) {
    if (self->hash_computed) {
        return self->hash;
    }

    Py_hash_t h = 0;
    PyObject *curr = (PyObject *)self;
    while (curr != Py_None && Py_TYPE(curr) == &ConsType) {
        Cons *c = (Cons *)curr;
        Py_hash_t item_hash = PyObject_Hash(c->first);
        if (item_hash == -1) {
            return -1;
        }
        h = 31 * h + item_hash;
        curr = c->rest;
    }

    self->hash = h;
    self->hash_computed = 1;
    return h;
}

static PyObject *Cons_richcompare(Cons *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self == (Cons *)other) {
        return PyBool_FromLong(op == Py_EQ);
    }

    if (!PyObject_TypeCheck(other, &ConsType)) {
        return PyBool_FromLong(op == Py_NE);
    }

    PyObject *a = (PyObject *)self;
    PyObject *b = other;

    while (a != Py_None && b != Py_None) {
        if (!PyObject_TypeCheck(a, &ConsType) || !PyObject_TypeCheck(b, &ConsType)) {
            break;
        }
        Cons *ca = (Cons *)a;
        Cons *cb = (Cons *)b;

        int cmp = PyObject_RichCompareBool(ca->first, cb->first, Py_EQ);
        if (cmp < 0) return NULL;
        if (!cmp) {
            return PyBool_FromLong(op == Py_NE);
        }
        a = ca->rest;
        b = cb->rest;
    }

    int both_none = (a == Py_None && b == Py_None);
    return PyBool_FromLong((op == Py_EQ) ? both_none : !both_none);
}

static PyObject *Cons_repr(Cons *self) {
    PyObject *parts = PyList_New(0);
    if (!parts) return NULL;

    PyObject *curr = (PyObject *)self;
    while (curr != Py_None && PyObject_TypeCheck(curr, &ConsType)) {
        Cons *c = (Cons *)curr;
        PyObject *repr = PyObject_Repr(c->first);
        if (!repr) {
            Py_DECREF(parts);
            return NULL;
        }
        if (PyList_Append(parts, repr) < 0) {
            Py_DECREF(repr);
            Py_DECREF(parts);
            return NULL;
        }
        Py_DECREF(repr);
        curr = c->rest;
    }

    PyObject *space = PyUnicode_FromString(" ");
    if (!space) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *joined = PyUnicode_Join(space, parts);
    Py_DECREF(space);
    Py_DECREF(parts);
    if (!joined) return NULL;

    PyObject *result = PyUnicode_FromFormat("(%U)", joined);
    Py_DECREF(joined);
    return result;
}

static PyObject *Cons_conj(Cons *self, PyObject *val) {
    Cons *new_cons = (Cons *)ConsType.tp_alloc(&ConsType, 0);
    if (!new_cons) return NULL;

    new_cons->first = val;
    Py_INCREF(val);
    new_cons->rest = (PyObject *)self;
    Py_INCREF(self);
    new_cons->hash = 0;
    new_cons->hash_computed = 0;

    return (PyObject *)new_cons;
}

static PyObject *Cons_reduce(Cons *self, PyObject *Py_UNUSED(ignored)) {
    // Convert Cons to a tuple of (first, rest)
    PyObject *args = PyTuple_Pack(2, self->first, self->rest ? self->rest : Py_None);
    if (args == NULL) {
        return NULL;
    }

    PyObject *result = PyTuple_Pack(2, (PyObject *)Py_TYPE(self), args);
    Py_DECREF(args);
    return result;
}

static PyMethodDef Cons_methods[] = {
    {"conj", (PyCFunction)Cons_conj, METH_O, "Add an element to the front"},
    {"__reduce__", (PyCFunction)Cons_reduce, METH_NOARGS, "Pickle support"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations (e.g., Cons[int])"},
    {NULL}
};

static PySequenceMethods Cons_as_sequence = {
    .sq_length = (lenfunc)Cons_length,
};

// Cons iterator
typedef struct {
    PyObject_HEAD
    PyObject *curr;
} ConsIterator;

static PyTypeObject ConsIteratorType;

static void ConsIterator_dealloc(ConsIterator *self) {
    Py_XDECREF(self->curr);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *ConsIterator_next(ConsIterator *self) {
    if (self->curr == Py_None || !PyObject_TypeCheck(self->curr, &ConsType)) {
        return NULL;  // StopIteration
    }

    Cons *c = (Cons *)self->curr;
    PyObject *result = c->first;
    Py_INCREF(result);

    PyObject *next = c->rest;
    Py_INCREF(next);
    Py_DECREF(self->curr);
    self->curr = next;

    return result;
}

static PyTypeObject ConsIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.ConsIterator",
    .tp_basicsize = sizeof(ConsIterator),
    .tp_dealloc = (destructor)ConsIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)ConsIterator_next,
};

static PyObject *Cons_iter(Cons *self) {
    ConsIterator *it = PyObject_New(ConsIterator, &ConsIteratorType);
    if (!it) return NULL;

    it->curr = (PyObject *)self;
    Py_INCREF(self);
    return (PyObject *)it;
}

static PyTypeObject ConsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.Cons",
    .tp_doc = "Immutable cons cell for persistent linked lists",
    .tp_basicsize = sizeof(Cons),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Cons_dealloc,
    .tp_repr = (reprfunc)Cons_repr,
    .tp_as_sequence = &Cons_as_sequence,
    .tp_hash = (hashfunc)Cons_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_richcompare = (richcmpfunc)Cons_richcompare,
    .tp_iter = (getiterfunc)Cons_iter,
    .tp_methods = Cons_methods,
    .tp_getset = Cons_getsetters,
    .tp_init = (initproc)Cons_init,
    .tp_new = Cons_new,
};

// === VectorNode ===
typedef struct VectorNode {
    PyObject_HEAD
    PyObject *array[WIDTH];
    PyObject *transient_id;
} VectorNode;

static PyTypeObject VectorNodeType;

static void VectorNode_dealloc(VectorNode *self) {
    for (int i = 0; i < WIDTH; i++) {
        Py_XDECREF(self->array[i]);
    }
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static VectorNode *VectorNode_create(PyObject *transient_id) {
    VectorNode *node = PyObject_New(VectorNode, &VectorNodeType);
    if (!node) return NULL;

    for (int i = 0; i < WIDTH; i++) {
        node->array[i] = NULL;
    }
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);
    return node;
}

static VectorNode *VectorNode_clone(VectorNode *self, PyObject *transient_id) {
    VectorNode *node = VectorNode_create(transient_id);
    if (!node) return NULL;

    for (int i = 0; i < WIDTH; i++) {
        node->array[i] = self->array[i];
        Py_XINCREF(node->array[i]);
    }
    return node;
}

static int VectorNode_is_editable(VectorNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

static PyTypeObject VectorNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.VectorNode",
    .tp_basicsize = sizeof(VectorNode),
    .tp_dealloc = (destructor)VectorNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// Global empty node
static VectorNode *EMPTY_NODE = NULL;

// === Vector ===
typedef struct Vector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    VectorNode *root;
    PyObject *tail;  // tuple
    Py_hash_t hash;
    int hash_computed;
    PyObject *transient_id;
    int initialized;
} Vector;

static PyTypeObject VectorType;
static Vector *EMPTY_VECTOR = NULL;

static void Vector_dealloc(Vector *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->tail);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Vector *self = (Vector *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->cnt = 0;
        self->shift = BITS;
        self->root = EMPTY_NODE;
        Py_INCREF(EMPTY_NODE);
        self->tail = PyTuple_New(0);
        self->hash = 0;
        self->hash_computed = 0;
        self->transient_id = NULL;
        self->initialized = 0;
    }
    return (PyObject *)self;
}

static Vector *Vector_create(Py_ssize_t cnt, int shift, VectorNode *root,
                               PyObject *tail, PyObject *transient_id) {
    Vector *vec = (Vector *)VectorType.tp_alloc(&VectorType, 0);
    if (!vec) return NULL;

    vec->cnt = cnt;
    vec->shift = shift;
    vec->root = root ? root : EMPTY_NODE;
    Py_INCREF(vec->root);
    vec->tail = tail ? tail : PyTuple_New(0);
    if (!vec->tail) {
        Py_DECREF(vec);
        return NULL;
    }
    if (tail) Py_INCREF(tail);
    vec->hash = 0;
    vec->hash_computed = 0;
    vec->transient_id = transient_id;
    Py_XINCREF(transient_id);
    vec->initialized = 1;

    return vec;
}

static Py_ssize_t Vector_length(Vector *self) {
    return self->cnt;
}

static Py_ssize_t Vector_tail_off(Vector *self) {
    if (self->cnt < WIDTH) {
        return 0;
    }
    return ((self->cnt - 1) >> BITS) << BITS;
}

// Internal helper: returns pointer to the raw array for index i
// For tail, returns NULL and sets *is_tail = 1, caller should use self->tail
// For tree nodes, returns the VectorNode* containing the array
// Does NOT create any new objects or do any refcounting
static VectorNode *Vector_node_for(Vector *self, Py_ssize_t i, int *is_tail) {
    *is_tail = 0;
    if (i >= Vector_tail_off(self)) {
        *is_tail = 1;
        return NULL;
    }

    VectorNode *node = self->root;
    for (int level = self->shift; level > 0; level -= BITS) {
        node = (VectorNode *)node->array[(i >> level) & MASK];
    }
    return node;
}

static PyObject *Vector_array_for(Vector *self, Py_ssize_t i) {
    if (i < 0 || i >= self->cnt) {
        PyErr_Format(PyExc_IndexError, "Index %zd out of range for vector of size %zd", i, self->cnt);
        return NULL;
    }

    if (i >= Vector_tail_off(self)) {
        Py_INCREF(self->tail);
        return self->tail;
    }

    VectorNode *node = self->root;
    for (int level = self->shift; level > 0; level -= BITS) {
        node = (VectorNode *)node->array[(i >> level) & MASK];
    }

    // Build a tuple from the node array
    PyObject *result = PyTuple_New(WIDTH);
    if (!result) return NULL;

    for (int j = 0; j < WIDTH; j++) {
        PyObject *item = node->array[j];
        if (item == NULL) item = Py_None;
        Py_INCREF(item);
        PyTuple_SET_ITEM(result, j, item);
    }
    return result;
}

// Internal C API - no argument parsing overhead
// Returns new reference, or NULL with exception set
static PyObject *Vector_nth_impl(Vector *self, Py_ssize_t i, PyObject *default_val) {
    if (i < 0) {
        i = self->cnt + i;
    }

    if (i < 0 || i >= self->cnt) {
        if (default_val != NULL) {
            Py_INCREF(default_val);
            return default_val;
        }
        PyErr_Format(PyExc_IndexError, "Index %zd out of range", i);
        return NULL;
    }

    PyObject *arr = Vector_array_for(self, i);
    if (!arr) return NULL;

    PyObject *result = PyTuple_GET_ITEM(arr, i & MASK);
    Py_INCREF(result);
    Py_DECREF(arr);
    return result;
}

// Python wrapper - parses arguments then calls impl
static PyObject *Vector_nth(Vector *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *default_val = NULL;

    if (!PyArg_ParseTuple(args, "n|O", &i, &default_val)) {
        return NULL;
    }

    return Vector_nth_impl(self, i, default_val);
}

static PyObject *Vector_getitem(Vector *self, PyObject *key) {
    if (PySlice_Check(key)) {
        // Handle slicing
        Py_ssize_t start, stop, step, slicelength;
        if (PySlice_GetIndicesEx(key, self->cnt, &start, &stop, &step, &slicelength) < 0) {
            return NULL;
        }

        // Build arguments for vec() function
        PyObject *items = PyList_New(slicelength);
        if (!items) return NULL;

        for (Py_ssize_t i = 0, j = start; i < slicelength; i++, j += step) {
            PyObject *arr = Vector_array_for(self, j);
            if (!arr) {
                Py_DECREF(items);
                return NULL;
            }
            PyObject *item = PyTuple_GET_ITEM(arr, j & MASK);
            Py_INCREF(item);
            PyList_SET_ITEM(items, i, item);
            Py_DECREF(arr);
        }

        // Create new Vector from list
        PyObject *result = PyObject_CallFunctionObjArgs((PyObject *)&VectorType, items, NULL);
        Py_DECREF(items);
        return result;
    }

    if (!PyLong_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
        return NULL;
    }

    Py_ssize_t i = PyLong_AsSsize_t(key);
    if (i == -1 && PyErr_Occurred()) return NULL;

    if (i < 0) {
        i = self->cnt + i;
    }

    if (i < 0 || i >= self->cnt) {
        PyErr_Format(PyExc_IndexError, "Index %zd out of range", i);
        return NULL;
    }

    PyObject *arr = Vector_array_for(self, i);
    if (!arr) return NULL;

    PyObject *result = PyTuple_GET_ITEM(arr, i & MASK);
    Py_INCREF(result);
    Py_DECREF(arr);
    return result;
}

static VectorNode *Vector_new_path(Vector *self, int level, VectorNode *node, PyObject *transient_id) {
    if (level == 0) {
        Py_INCREF(node);
        return node;
    }
    VectorNode *ret = VectorNode_create(transient_id);
    if (!ret) return NULL;

    VectorNode *child = Vector_new_path(self, level - BITS, node, transient_id);
    if (!child) {
        Py_DECREF(ret);
        return NULL;
    }
    ret->array[0] = (PyObject *)child;
    return ret;
}

static VectorNode *Vector_push_tail(Vector *self, int level, VectorNode *parent, VectorNode *tail_node, PyObject *transient_id) {
    int subidx = ((self->cnt - 1) >> level) & MASK;
    VectorNode *ret;

    if (VectorNode_is_editable(parent, transient_id)) {
        ret = parent;
        Py_INCREF(ret);
    } else {
        ret = VectorNode_clone(parent, transient_id);
        if (!ret) return NULL;
    }

    VectorNode *node_to_insert;
    if (level == BITS) {
        node_to_insert = tail_node;
        Py_INCREF(tail_node);
    } else {
        PyObject *child = parent->array[subidx];
        if (child != NULL) {
            node_to_insert = Vector_push_tail(self, level - BITS, (VectorNode *)child, tail_node, transient_id);
        } else {
            node_to_insert = Vector_new_path(self, level - BITS, tail_node, transient_id);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    Py_XDECREF(ret->array[subidx]);
    ret->array[subidx] = (PyObject *)node_to_insert;
    return ret;
}

static PyObject *Vector_conj(Vector *self, PyObject *val) {
    PyObject *transient_id = self->transient_id;

    // Room in tail?
    Py_ssize_t tail_len = PyTuple_Size(self->tail);
    if (self->cnt - Vector_tail_off(self) < WIDTH) {
        PyObject *new_tail = PyTuple_New(tail_len + 1);
        if (!new_tail) return NULL;

        for (Py_ssize_t i = 0; i < tail_len; i++) {
            PyObject *item = PyTuple_GET_ITEM(self->tail, i);
            Py_INCREF(item);
            PyTuple_SET_ITEM(new_tail, i, item);
        }
        Py_INCREF(val);
        PyTuple_SET_ITEM(new_tail, tail_len, val);

        Vector *result = Vector_create(self->cnt + 1, self->shift, self->root, new_tail, transient_id);
        Py_DECREF(new_tail);
        return (PyObject *)result;
    }

    // Tail is full, push into trie
    VectorNode *tail_node = VectorNode_create(transient_id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < tail_len && i < WIDTH; i++) {
        tail_node->array[i] = PyTuple_GET_ITEM(self->tail, i);
        Py_INCREF(tail_node->array[i]);
    }

    int new_shift = self->shift;
    VectorNode *new_root;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        new_root = VectorNode_create(transient_id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->array[0] = (PyObject *)self->root;
        Py_INCREF(self->root);

        VectorNode *path = Vector_new_path(self, self->shift, tail_node, transient_id);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->array[1] = (PyObject *)path;
        new_shift += BITS;
    } else {
        new_root = Vector_push_tail(self, self->shift, self->root, tail_node, transient_id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
    }

    Py_DECREF(tail_node);

    PyObject *new_tail = PyTuple_New(1);
    if (!new_tail) {
        Py_DECREF(new_root);
        return NULL;
    }
    Py_INCREF(val);
    PyTuple_SET_ITEM(new_tail, 0, val);

    Vector *result = Vector_create(self->cnt + 1, new_shift, new_root, new_tail, transient_id);
    Py_DECREF(new_root);
    Py_DECREF(new_tail);
    return (PyObject *)result;
}

static VectorNode *Vector_do_assoc(Vector *self, int level, VectorNode *node, Py_ssize_t i, PyObject *val) {
    VectorNode *ret = VectorNode_clone(node, NULL);
    if (!ret) return NULL;

    if (level == 0) {
        Py_XDECREF(ret->array[i & MASK]);
        ret->array[i & MASK] = val;
        Py_INCREF(val);
    } else {
        int subidx = (i >> level) & MASK;
        VectorNode *child = Vector_do_assoc(self, level - BITS, (VectorNode *)node->array[subidx], i, val);
        if (!child) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_XDECREF(ret->array[subidx]);
        ret->array[subidx] = (PyObject *)child;
    }
    return ret;
}

static PyObject *Vector_assoc(Vector *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *val;

    if (!PyArg_ParseTuple(args, "nO", &i, &val)) {
        return NULL;
    }

    if (i < 0) {
        i = self->cnt + i;
    }

    if (i < 0 || i > self->cnt) {
        PyErr_Format(PyExc_IndexError, "Index %zd out of range", i);
        return NULL;
    }

    if (i == self->cnt) {
        return Vector_conj(self, val);
    }

    if (i >= Vector_tail_off(self)) {
        // Update in tail
        Py_ssize_t tail_len = PyTuple_Size(self->tail);
        PyObject *new_tail = PyTuple_New(tail_len);
        if (!new_tail) return NULL;

        for (Py_ssize_t j = 0; j < tail_len; j++) {
            PyObject *item;
            if (j == (i & MASK)) {
                item = val;
            } else {
                item = PyTuple_GET_ITEM(self->tail, j);
            }
            Py_INCREF(item);
            PyTuple_SET_ITEM(new_tail, j, item);
        }

        Vector *result = Vector_create(self->cnt, self->shift, self->root, new_tail, NULL);
        Py_DECREF(new_tail);
        return (PyObject *)result;
    }

    // Update in trie
    VectorNode *new_root = Vector_do_assoc(self, self->shift, self->root, i, val);
    if (!new_root) return NULL;

    Vector *result = Vector_create(self->cnt, self->shift, new_root, self->tail, NULL);
    Py_DECREF(new_root);
    return (PyObject *)result;
}

static VectorNode *Vector_pop_tail(Vector *self, int level, VectorNode *node) {
    int subidx = ((self->cnt - 2) >> level) & MASK;

    if (level > BITS) {
        VectorNode *new_child = Vector_pop_tail(self, level - BITS, (VectorNode *)node->array[subidx]);
        if (new_child == NULL && subidx == 0) {
            return NULL;
        }
        VectorNode *ret = VectorNode_clone(node, NULL);
        if (!ret) return NULL;
        Py_XDECREF(ret->array[subidx]);
        ret->array[subidx] = (PyObject *)new_child;
        return ret;
    } else if (subidx == 0) {
        return NULL;
    } else {
        VectorNode *ret = VectorNode_clone(node, NULL);
        if (!ret) return NULL;
        Py_XDECREF(ret->array[subidx]);
        ret->array[subidx] = NULL;
        return ret;
    }
}

static PyObject *Vector_pop(Vector *self, PyObject *Py_UNUSED(ignored)) {
    if (self->cnt == 0) {
        PyErr_SetString(PyExc_IndexError, "Can't pop empty vector");
        return NULL;
    }
    if (self->cnt == 1) {
        Py_INCREF(EMPTY_VECTOR);
        return (PyObject *)EMPTY_VECTOR;
    }

    // More than one in tail?
    Py_ssize_t tail_len = PyTuple_Size(self->tail);
    if (self->cnt - Vector_tail_off(self) > 1) {
        PyObject *new_tail = PyTuple_GetSlice(self->tail, 0, tail_len - 1);
        if (!new_tail) return NULL;

        Vector *result = Vector_create(self->cnt - 1, self->shift, self->root, new_tail, NULL);
        Py_DECREF(new_tail);
        return (PyObject *)result;
    }

    // Pop from trie
    PyObject *new_tail = Vector_array_for(self, self->cnt - 2);
    if (!new_tail) return NULL;

    VectorNode *new_root = Vector_pop_tail(self, self->shift, self->root);
    int new_shift = self->shift;

    if (new_root == NULL) {
        new_root = EMPTY_NODE;
        Py_INCREF(new_root);
    }

    if (self->shift > BITS && new_root->array[1] == NULL) {
        VectorNode *old_root = new_root;
        new_root = (VectorNode *)new_root->array[0];
        Py_INCREF(new_root);
        Py_DECREF(old_root);
        new_shift -= BITS;
    }

    Vector *result = Vector_create(self->cnt - 1, new_shift, new_root, new_tail, NULL);
    Py_DECREF(new_root);
    Py_DECREF(new_tail);
    return (PyObject *)result;
}

// Forward declaration for TransientVector
static PyTypeObject TransientVectorType;
static PyObject *Vector_transient(Vector *self, PyObject *Py_UNUSED(ignored));
static PyObject *TransientVector_conj_mut(TransientVector *self, PyObject *val);
static PyObject *TransientVector_persistent(TransientVector *self, PyObject *Py_UNUSED(ignored));

static PyObject *Vector_add(PyObject *left, PyObject *right) {
    // nb_add can be called for reflected operations. Validate the left operand
    // before treating it as a Vector; iterable + Vector is intentionally not
    // supported.
    if (!PyObject_TypeCheck(left, &VectorType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    Vector *self = (Vector *)left;

    // Try to get an iterator - this handles any iterable including Vector.
    PyObject *iter = PyObject_GetIter(right);
    if (!iter) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        return NULL;
    }

    // Create transient directly via C function call (no Python method lookup).
    TransientVector *t = (TransientVector *)Vector_transient(self, NULL);
    if (!t) {
        Py_DECREF(iter);
        return NULL;
    }

    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        PyObject *res = TransientVector_conj_mut(t, item);
        Py_DECREF(item);
        if (!res) {
            Py_DECREF(iter);
            Py_DECREF(t);
            return NULL;
        }
        Py_DECREF(res);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        Py_DECREF(t);
        return NULL;
    }

    PyObject *result = TransientVector_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}

static PyObject *Vector_multiply(PyObject *left, PyObject *right) {
    Vector *self;
    PyObject *count_obj;

    // Multiplication is commutative for repetition: support both v * n and
    // n * v while still validating reflected slot calls safely.
    if (PyObject_TypeCheck(left, &VectorType)) {
        self = (Vector *)left;
        count_obj = right;
    } else if (PyObject_TypeCheck(right, &VectorType)) {
        self = (Vector *)right;
        count_obj = left;
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }

    PyObject *index = PyNumber_Index(count_obj);
    if (!index) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        return NULL;
    }

    Py_ssize_t count = PyLong_AsSsize_t(index);
    Py_DECREF(index);
    if (count == -1 && PyErr_Occurred()) {
        return NULL;
    }

    if (count <= 0 || self->cnt == 0) {
        Py_INCREF(EMPTY_VECTOR);
        return (PyObject *)EMPTY_VECTOR;
    }
    if (count == 1) {
        Py_INCREF(self);
        return (PyObject *)self;
    }
    if (self->cnt > PY_SSIZE_T_MAX / count) {
        PyErr_SetString(PyExc_OverflowError, "repeated Vector is too long");
        return NULL;
    }

    TransientVector *t = (TransientVector *)Vector_transient(EMPTY_VECTOR, NULL);
    if (!t) return NULL;

    for (Py_ssize_t repetition = 0; repetition < count; repetition++) {
        PyObject *iter = PyObject_GetIter((PyObject *)self);
        if (!iter) {
            Py_DECREF(t);
            return NULL;
        }

        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            PyObject *res = TransientVector_conj_mut(t, item);
            Py_DECREF(item);
            if (!res) {
                Py_DECREF(iter);
                Py_DECREF(t);
                return NULL;
            }
            Py_DECREF(res);
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) {
            Py_DECREF(t);
            return NULL;
        }
    }

    PyObject *result = TransientVector_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}

static Py_hash_t Vector_hash(Vector *self) {
    if (self->hash_computed) {
        return self->hash;
    }

    Py_hash_t h = 0;
    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        PyObject *arr = Vector_array_for(self, i);
        if (!arr) return -1;

        PyObject *item = PyTuple_GET_ITEM(arr, i & MASK);
        Py_hash_t item_hash = PyObject_Hash(item);
        Py_DECREF(arr);

        if (item_hash == -1) return -1;
        h = 31 * h + item_hash;
    }

    self->hash = h;
    self->hash_computed = 1;
    return h;
}

static PyObject *Vector_richcompare(Vector *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self == (Vector *)other) {
        return PyBool_FromLong(op == Py_EQ);
    }

    if (!PyObject_TypeCheck(other, &VectorType)) {
        return PyBool_FromLong(op == Py_NE);
    }

    Vector *o = (Vector *)other;
    if (self->cnt != o->cnt) {
        return PyBool_FromLong(op == Py_NE);
    }

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        PyObject *arr1 = Vector_array_for(self, i);
        PyObject *arr2 = Vector_array_for(o, i);
        if (!arr1 || !arr2) {
            Py_XDECREF(arr1);
            Py_XDECREF(arr2);
            return NULL;
        }

        int cmp = PyObject_RichCompareBool(PyTuple_GET_ITEM(arr1, i & MASK),
                                           PyTuple_GET_ITEM(arr2, i & MASK), Py_EQ);
        Py_DECREF(arr1);
        Py_DECREF(arr2);

        if (cmp < 0) return NULL;
        if (!cmp) {
            return PyBool_FromLong(op == Py_NE);
        }
    }

    return PyBool_FromLong(op == Py_EQ);
}

static PyObject *Vector_repr(Vector *self) {
    PyObject *parts = PyList_New(0);
    if (!parts) return NULL;

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        PyObject *arr = Vector_array_for(self, i);
        if (!arr) {
            Py_DECREF(parts);
            return NULL;
        }

        PyObject *item = PyTuple_GET_ITEM(arr, i & MASK);
        PyObject *repr = PyObject_Repr(item);
        Py_DECREF(arr);

        if (!repr) {
            Py_DECREF(parts);
            return NULL;
        }
        if (PyList_Append(parts, repr) < 0) {
            Py_DECREF(repr);
            Py_DECREF(parts);
            return NULL;
        }
        Py_DECREF(repr);
    }

    PyObject *space = PyUnicode_FromString(" ");
    if (!space) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *joined = PyUnicode_Join(space, parts);
    Py_DECREF(space);
    Py_DECREF(parts);
    if (!joined) return NULL;

    PyObject *result = PyUnicode_FromFormat("[%U]", joined);
    Py_DECREF(joined);
    return result;
}


static PyObject *Vector_to_seq(Vector *self, PyObject *Py_UNUSED(ignored)) {
    if (self->cnt == 0) {
        Py_RETURN_NONE;
    }

    // Build Cons list in reverse, processing chunks at a time
    // Uses Vector_node_for to avoid creating intermediate tuples
    Cons *result = NULL;
    Py_ssize_t i = self->cnt - 1;

    while (i >= 0) {
        // Determine the chunk boundaries for this index
        Py_ssize_t chunk_start = (i >> BITS) << BITS;

        // Get direct access to the array for this chunk (no tuple allocation)
        int is_tail;
        VectorNode *node = Vector_node_for(self, i, &is_tail);

        // Process all elements in this chunk, from i down to chunk_start
        for (Py_ssize_t j = i; j >= chunk_start; j--) {
            PyObject *item;
            if (is_tail) {
                item = PyTuple_GET_ITEM(self->tail, j & MASK);
            } else {
                item = node->array[j & MASK];
                if (item == NULL) item = Py_None;
            }

            Cons *new_cons = (Cons *)ConsType.tp_alloc(&ConsType, 0);
            if (!new_cons) {
                Py_XDECREF(result);
                return NULL;
            }

            Py_INCREF(item);
            new_cons->first = item;
            new_cons->rest = result ? (PyObject *)result : Py_None;
            Py_INCREF(new_cons->rest);
            new_cons->hash = 0;
            new_cons->hash_computed = 0;

            result = new_cons;
        }

        // Move to the last element of the previous chunk
        i = chunk_start - 1;
    }

    return (PyObject *)result;
}

/* Vector.copy() - returns self since Vector is immutable */
static PyObject *Vector_copy(Vector *self, PyObject *Py_UNUSED(ignored)) {
    Py_INCREF(self);
    return (PyObject *)self;
}

/* Vector.sort(key=None, reverse=False) - return a new sorted vector */
static PyObject *Vector_sort(Vector *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t len = self->cnt;
    if (len <= 1) {
        Py_INCREF(self);
        return (PyObject *)self;  // Already sorted, return self
    }

    // Create a transient from this vector
    PyObject *transient = Vector_transient(self, NULL);
    if (!transient) return NULL;

    // Call sort on the transient
    PyObject *sort_method = PyObject_GetAttrString(transient, "sort");
    if (!sort_method) {
        Py_DECREF(transient);
        return NULL;
    }

    PyObject *empty_args = PyTuple_New(0);
    if (!empty_args) {
        Py_DECREF(sort_method);
        Py_DECREF(transient);
        return NULL;
    }

    PyObject *sort_result = PyObject_Call(sort_method, empty_args, kwargs);
    Py_DECREF(empty_args);
    Py_DECREF(sort_method);
    if (!sort_result) {
        Py_DECREF(transient);
        return NULL;
    }
    Py_DECREF(sort_result);

    // Convert back to persistent
    PyObject *persistent_method = PyObject_GetAttrString(transient, "persistent");
    if (!persistent_method) {
        Py_DECREF(transient);
        return NULL;
    }

    PyObject *result = PyObject_CallNoArgs(persistent_method);
    Py_DECREF(persistent_method);
    Py_DECREF(transient);

    return result;
}

/* Vector.index(value, start=0, stop=len) - find index of value */
static PyObject *Vector_index(Vector *self, PyObject *args) {
    PyObject *value;
    Py_ssize_t start = 0;
    Py_ssize_t stop = self->cnt;

    if (!PyArg_ParseTuple(args, "O|nn", &value, &start, &stop)) {
        return NULL;
    }

    if (start < 0) {
        start += self->cnt;
        if (start < 0) start = 0;
    }
    if (stop < 0) {
        stop += self->cnt;
    }
    if (stop > self->cnt) {
        stop = self->cnt;
    }

    for (Py_ssize_t i = start; i < stop; i++) {
        PyObject *item = Vector_nth_impl(self, i, NULL);
        if (!item) return NULL;

        int cmp = PyObject_RichCompareBool(item, value, Py_EQ);
        Py_DECREF(item);

        if (cmp < 0) return NULL;  // Error
        if (cmp == 1) {
            return PyLong_FromSsize_t(i);
        }
    }

    PyErr_SetString(PyExc_ValueError, "value not in vector");
    return NULL;
}

/* Vector.count(value) - count occurrences of value */
static PyObject *Vector_count(Vector *self, PyObject *value) {
    Py_ssize_t count = 0;

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        PyObject *item = Vector_nth_impl(self, i, NULL);
        if (!item) return NULL;

        int cmp = PyObject_RichCompareBool(item, value, Py_EQ);
        Py_DECREF(item);

        if (cmp < 0) return NULL;  // Error
        if (cmp == 1) count++;
    }

    return PyLong_FromSsize_t(count);
}

static PyObject *Vector_reduce(Vector *self, PyObject *Py_UNUSED(ignored)) {
    // Vector accepts one iterable constructor argument, so wrap the elements
    // tuple in the argument tuple used by pickle.
    PyObject *elements = PySequence_Tuple((PyObject *)self);
    if (!elements) return NULL;

    PyObject *args = PyTuple_Pack(1, elements);
    Py_DECREF(elements);
    if (!args) return NULL;

    PyObject *result = PyTuple_Pack(2, (PyObject *)Py_TYPE(self), args);
    Py_DECREF(args);
    return result;
}

static PyMethodDef Vector_methods[] = {
    {"nth", (PyCFunction)Vector_nth, METH_VARARGS, "Get element at index"},
    {"conj", (PyCFunction)Vector_conj, METH_O, "Add element to end"},
    {"assoc", (PyCFunction)Vector_assoc, METH_VARARGS, "Set element at index"},
    {"pop", (PyCFunction)Vector_pop, METH_NOARGS, "Remove last element"},
    {"transient", (PyCFunction)Vector_transient, METH_NOARGS, "Get transient version"},
    {"to_seq", (PyCFunction)Vector_to_seq, METH_NOARGS, "Convert to Cons sequence"},
    {"copy", (PyCFunction)Vector_copy, METH_NOARGS, "Return self (immutable vectors don't need copying)"},
    {"index", (PyCFunction)Vector_index, METH_VARARGS, "Return index of first occurrence of value"},
    {"count", (PyCFunction)Vector_count, METH_O, "Return number of occurrences of value"},
    {"sort", (PyCFunction)Vector_sort, METH_VARARGS | METH_KEYWORDS, "Return a new sorted vector"},
    {"__reduce__", (PyCFunction)Vector_reduce, METH_NOARGS, "Pickle support"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations (e.g., Vector[int])"},
    {NULL}
};

static PyMappingMethods Vector_as_mapping = {
    .mp_length = (lenfunc)Vector_length,
    .mp_subscript = (binaryfunc)Vector_getitem,
};

// Sequence protocol sq_item - takes Py_ssize_t index directly
static PyObject *Vector_sq_item(Vector *self, Py_ssize_t i) {
    if (i < 0) {
        i = self->cnt + i;
    }
    if (i < 0 || i >= self->cnt) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    PyObject *arr = Vector_array_for(self, i);
    if (!arr) return NULL;

    PyObject *result = PyTuple_GET_ITEM(arr, i & MASK);
    Py_INCREF(result);
    Py_DECREF(arr);
    return result;
}

static PySequenceMethods Vector_as_sequence = {
    .sq_length = (lenfunc)Vector_length,
    .sq_item = (ssizeargfunc)Vector_sq_item,
};

static PyNumberMethods Vector_as_number = {
    .nb_add = Vector_add,
    .nb_multiply = Vector_multiply,
};

// Vector iterator
typedef struct {
    PyObject_HEAD
    Vector *vec;
    Py_ssize_t index;
    VectorNode *cached_node;   // Cached tree node (NULL if in tail)
    Py_ssize_t cached_chunk;   // Which chunk is cached (chunk_start index)
    int cached_is_tail;        // Whether cached chunk is the tail
} VectorIterator;

static PyTypeObject VectorIteratorType;

static void VectorIterator_dealloc(VectorIterator *self) {
    Py_XDECREF(self->vec);
    Py_XDECREF(self->cached_node);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *VectorIterator_next(VectorIterator *self) {
    if (self->index >= self->vec->cnt) {
        return NULL;
    }

    // Check if we need to fetch a new chunk
    Py_ssize_t chunk_start = (self->index >> BITS) << BITS;
    if (chunk_start != self->cached_chunk) {
        Py_XDECREF(self->cached_node);
        self->cached_node = Vector_node_for(self->vec, self->index, &self->cached_is_tail);
        Py_XINCREF(self->cached_node);
        self->cached_chunk = chunk_start;
    }

    PyObject *result;
    if (self->cached_is_tail) {
        result = PyTuple_GET_ITEM(self->vec->tail, self->index & MASK);
    } else {
        result = self->cached_node->array[self->index & MASK];
        if (result == NULL) result = Py_None;
    }
    Py_INCREF(result);

    self->index++;
    return result;
}

static PyTypeObject VectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.VectorIterator",
    .tp_basicsize = sizeof(VectorIterator),
    .tp_dealloc = (destructor)VectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)VectorIterator_next,
};

static PyObject *Vector_iter(Vector *self) {
    VectorIterator *it = PyObject_New(VectorIterator, &VectorIteratorType);
    if (!it) return NULL;

    it->vec = self;
    Py_INCREF(self);
    it->index = 0;
    it->cached_node = NULL;
    it->cached_chunk = -1;  // Invalid chunk to force initial fetch
    it->cached_is_tail = 0;
    return (PyObject *)it;
}

static int Vector_init(Vector *self, PyObject *args, PyObject *kwds) {
    if (self->initialized) {
        PyErr_SetString(PyExc_TypeError, "Vector values cannot be reinitialized");
        return -1;
    }

    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (kwds && PyDict_GET_SIZE(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "Vector takes no keyword arguments");
        return -1;
    }
    if (nargs > 1) {
        PyErr_Format(
            PyExc_TypeError,
            "Vector expected at most 1 argument, got %zd",
            nargs
        );
        return -1;
    }
    if (nargs == 0) {
        self->initialized = 1;
        return 0;
    }

    PyObject *iter = PyObject_GetIter(PyTuple_GET_ITEM(args, 0));
    if (!iter) return -1;

    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        PyObject *new_vec = Vector_conj(self, item);
        Py_DECREF(item);
        if (!new_vec) {
            Py_DECREF(iter);
            return -1;
        }

        Vector *nv = (Vector *)new_vec;
        Py_DECREF(self->root);
        Py_DECREF(self->tail);
        self->cnt = nv->cnt;
        self->shift = nv->shift;
        self->root = nv->root;
        Py_INCREF(self->root);
        self->tail = nv->tail;
        Py_INCREF(self->tail);
        Py_DECREF(new_vec);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) return -1;
    self->initialized = 1;
    return 0;
}

static PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.Vector",
    .tp_doc = "Vector(iterable=()) -> persistent vector using a bit-partitioned trie",
    .tp_basicsize = sizeof(Vector),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Vector_dealloc,
    .tp_repr = (reprfunc)Vector_repr,
    .tp_as_number = &Vector_as_number,
    .tp_as_sequence = &Vector_as_sequence,
    .tp_as_mapping = &Vector_as_mapping,
    .tp_hash = (hashfunc)Vector_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_richcompare = (richcmpfunc)Vector_richcompare,
    .tp_iter = (getiterfunc)Vector_iter,
    .tp_methods = Vector_methods,
    .tp_init = (initproc)Vector_init,
    .tp_new = Vector_new,
};

// =============================================================================
// DOUBLEVECTOR - Type-specialized vector for double (float64)
// =============================================================================

// Forward declarations for DoubleVector
static PyObject *DoubleVector_conj(DoubleVector *self, PyObject *val);

// --- DoubleVectorNode ---
// For internal nodes: array stores pointers to child nodes (cast to void*)
// For leaf nodes: array stores double values
// We use a union to avoid strict-aliasing issues
typedef struct DoubleVectorNode {
    PyObject_HEAD
    union {
        double values[WIDTH];
        struct DoubleVectorNode *children[WIDTH];
    } data;
    int valid_mask;  // Bitmask of which slots are valid
    PyObject *transient_id;
} DoubleVectorNode;

static PyTypeObject DoubleVectorNodeType;

static void DoubleVectorNode_dealloc(DoubleVectorNode *self) {
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static DoubleVectorNode *DoubleVectorNode_create(PyObject *transient_id) {
    DoubleVectorNode *node = PyObject_New(DoubleVectorNode, &DoubleVectorNodeType);
    if (!node) return NULL;

    for (int i = 0; i < WIDTH; i++) {
        node->data.values[i] = 0.0;
        node->data.children[i] = NULL;
    }
    node->valid_mask = 0;
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);
    return node;
}

static DoubleVectorNode *DoubleVectorNode_clone(DoubleVectorNode *self, PyObject *transient_id) {
    DoubleVectorNode *node = DoubleVectorNode_create(transient_id);
    if (!node) return NULL;

    // Copy the entire union
    memcpy(&node->data, &self->data, sizeof(self->data));
    node->valid_mask = self->valid_mask;
    return node;
}

static int DoubleVectorNode_is_editable(DoubleVectorNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

static PyTypeObject DoubleVectorNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.DoubleVectorNode",
    .tp_basicsize = sizeof(DoubleVectorNode),
    .tp_dealloc = (destructor)DoubleVectorNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// Global empty double node
static DoubleVectorNode *EMPTY_DOUBLE_NODE = NULL;

// --- DoubleVector ---
typedef struct DoubleVector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    DoubleVectorNode *root;
    double *tail;
    Py_ssize_t tail_len;
    Py_ssize_t tail_cap;
    Py_hash_t hash;
    int hash_computed;
    PyObject *transient_id;
    // Buffer protocol cache
    double *flat_buffer_cache;
} DoubleVector;

static PyTypeObject DoubleVectorType;
static DoubleVector *EMPTY_DOUBLE_VECTOR = NULL;

static void DoubleVector_dealloc(DoubleVector *self) {
    Py_XDECREF(self->root);
    if (self->tail) free(self->tail);
    if (self->flat_buffer_cache) free(self->flat_buffer_cache);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DoubleVector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    DoubleVector *self = (DoubleVector *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->cnt = 0;
        self->shift = BITS;
        self->root = EMPTY_DOUBLE_NODE;
        Py_INCREF(EMPTY_DOUBLE_NODE);
        self->tail = NULL;
        self->tail_len = 0;
        self->tail_cap = 0;
        self->hash = 0;
        self->hash_computed = 0;
        self->transient_id = NULL;
        self->flat_buffer_cache = NULL;
    }
    return (PyObject *)self;
}

static DoubleVector *DoubleVector_create(Py_ssize_t cnt, int shift, DoubleVectorNode *root,
                                           double *tail, Py_ssize_t tail_len, PyObject *transient_id) {
    DoubleVector *vec = (DoubleVector *)DoubleVectorType.tp_alloc(&DoubleVectorType, 0);
    if (!vec) return NULL;

    vec->cnt = cnt;
    vec->shift = shift;
    vec->root = root ? root : EMPTY_DOUBLE_NODE;
    Py_INCREF(vec->root);

    if (tail && tail_len > 0) {
        vec->tail = (double *)malloc(tail_len * sizeof(double));
        if (!vec->tail) {
            Py_DECREF(vec);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(vec->tail, tail, tail_len * sizeof(double));
        vec->tail_len = tail_len;
        vec->tail_cap = tail_len;
    } else {
        vec->tail = NULL;
        vec->tail_len = 0;
        vec->tail_cap = 0;
    }

    vec->hash = 0;
    vec->hash_computed = 0;
    vec->transient_id = transient_id;
    Py_XINCREF(transient_id);
    vec->flat_buffer_cache = NULL;

    return vec;
}

static Py_ssize_t DoubleVector_length(DoubleVector *self) {
    return self->cnt;
}

static Py_ssize_t DoubleVector_tail_off(DoubleVector *self) {
    if (self->cnt < WIDTH) {
        return 0;
    }
    return ((self->cnt - 1) >> BITS) << BITS;
}

// Get the leaf node array for index i (returns pointer to node's values array)
static double *DoubleVector_array_for(DoubleVector *self, Py_ssize_t i) {
    if (i < 0 || i >= self->cnt) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    if (i >= DoubleVector_tail_off(self)) {
        return self->tail;
    }

    DoubleVectorNode *node = self->root;
    for (int level = self->shift; level > 0; level -= BITS) {
        int idx = (i >> level) & MASK;
        node = node->data.children[idx];
    }
    return node->data.values;
}

// Get raw double at index (no boxing)
static double DoubleVector_nth_raw(DoubleVector *self, Py_ssize_t i) {
    double *arr = DoubleVector_array_for(self, i);
    if (!arr) return 0.0;  // Error already set
    return arr[i & MASK];
}

// Get element at index, boxed as PyObject
static PyObject *DoubleVector_nth(DoubleVector *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *default_val = NULL;

    if (!PyArg_ParseTuple(args, "n|O", &i, &default_val)) {
        return NULL;
    }

    if (i < 0) {
        i = self->cnt + i;
    }

    if (i < 0 || i >= self->cnt) {
        if (default_val) {
            Py_INCREF(default_val);
            return default_val;
        }
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    double val = DoubleVector_nth_raw(self, i);
    if (PyErr_Occurred()) return NULL;
    return PyFloat_FromDouble(val);
}

static PyObject *DoubleVector_getitem(DoubleVector *self, PyObject *key) {
    if (PyIndex_Check(key)) {
        Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred()) return NULL;

        if (i < 0) i = self->cnt + i;
        if (i < 0 || i >= self->cnt) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        double val = DoubleVector_nth_raw(self, i);
        if (PyErr_Occurred()) return NULL;
        return PyFloat_FromDouble(val);
    }

    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;
        if (PySlice_GetIndicesEx(key, self->cnt, &start, &stop, &step, &slicelength) < 0) {
            return NULL;
        }

        // Create new double vector from slice
        DoubleVector *result = DoubleVector_create(0, BITS, NULL, NULL, 0, NULL);
        if (!result) return NULL;

        for (Py_ssize_t i = start, j = 0; j < slicelength; i += step, j++) {
            double val = DoubleVector_nth_raw(self, i);
            if (PyErr_Occurred()) {
                Py_DECREF(result);
                return NULL;
            }
            // Use conj to add - need to implement
            PyObject *boxed = PyFloat_FromDouble(val);
            if (!boxed) {
                Py_DECREF(result);
                return NULL;
            }
            PyObject *new_result = DoubleVector_conj(result, boxed);
            Py_DECREF(boxed);
            Py_DECREF(result);
            if (!new_result) return NULL;
            result = (DoubleVector *)new_result;
        }

        return (PyObject *)result;
    }

    PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
    return NULL;
}

static DoubleVectorNode *DoubleVector_new_path(DoubleVector *self, int level, DoubleVectorNode *node, PyObject *transient_id) {
    if (level == 0) {
        Py_INCREF(node);
        return node;
    }
    DoubleVectorNode *ret = DoubleVectorNode_create(transient_id);
    if (!ret) return NULL;

    DoubleVectorNode *child = DoubleVector_new_path(self, level - BITS, node, transient_id);
    if (!child) {
        Py_DECREF(ret);
        return NULL;
    }
    ret->data.children[0] = child;
    ret->valid_mask = 1;
    return ret;
}

static DoubleVectorNode *DoubleVector_push_tail(DoubleVector *self, int level, DoubleVectorNode *parent, DoubleVectorNode *tail_node, PyObject *transient_id) {
    int subidx = ((self->cnt - 1) >> level) & MASK;
    DoubleVectorNode *ret;

    if (DoubleVectorNode_is_editable(parent, transient_id)) {
        ret = parent;
        Py_INCREF(ret);
    } else {
        ret = DoubleVectorNode_clone(parent, transient_id);
        if (!ret) return NULL;
    }

    DoubleVectorNode *node_to_insert;
    if (level == BITS) {
        node_to_insert = tail_node;
        Py_INCREF(tail_node);
    } else {
        DoubleVectorNode *child = parent->data.children[subidx];
        if (child != NULL && (parent->valid_mask & (1 << subidx))) {
            node_to_insert = DoubleVector_push_tail(self, level - BITS, child, tail_node, transient_id);
        } else {
            node_to_insert = DoubleVector_new_path(self, level - BITS, tail_node, transient_id);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    // Clean up old child if present
    if (ret->valid_mask & (1 << subidx)) {
        Py_XDECREF(ret->data.children[subidx]);
    }
    ret->data.children[subidx] = node_to_insert;
    ret->valid_mask |= (1 << subidx);
    return ret;
}

static PyObject *DoubleVector_conj(DoubleVector *self, PyObject *val) {
    // Unbox the value
    double dval = PyFloat_AsDouble(val);
    if (dval == -1.0 && PyErr_Occurred()) {
        return NULL;
    }

    PyObject *transient_id = self->transient_id;

    // Room in tail?
    if (self->cnt - DoubleVector_tail_off(self) < WIDTH) {
        Py_ssize_t new_tail_len = self->tail_len + 1;
        double *new_tail = (double *)malloc(new_tail_len * sizeof(double));
        if (!new_tail) {
            PyErr_NoMemory();
            return NULL;
        }

        if (self->tail && self->tail_len > 0) {
            memcpy(new_tail, self->tail, self->tail_len * sizeof(double));
        }
        new_tail[self->tail_len] = dval;

        DoubleVector *result = DoubleVector_create(self->cnt + 1, self->shift, self->root,
                                                      new_tail, new_tail_len, transient_id);
        free(new_tail);
        return (PyObject *)result;
    }

    // Tail is full, push into trie
    DoubleVectorNode *tail_node = DoubleVectorNode_create(transient_id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1 << i);
    }

    int new_shift = self->shift;
    DoubleVectorNode *new_root;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        new_root = DoubleVectorNode_create(transient_id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[0] = self->root;
        Py_INCREF(self->root);
        new_root->valid_mask = 1;

        DoubleVectorNode *path = DoubleVector_new_path(self, self->shift, tail_node, transient_id);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[1] = path;
        new_root->valid_mask |= 2;
        new_shift += BITS;
    } else {
        new_root = DoubleVector_push_tail(self, self->shift, self->root, tail_node, transient_id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
    }

    Py_DECREF(tail_node);

    double new_tail_arr[1] = { dval };
    DoubleVector *result = DoubleVector_create(self->cnt + 1, new_shift, new_root,
                                                  new_tail_arr, 1, transient_id);
    Py_DECREF(new_root);
    return (PyObject *)result;
}

static PyObject *DoubleVector_repr(DoubleVector *self) {
    PyObject *result = PyUnicode_FromString("vec_f64([");
    if (!result) return NULL;

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        double val = DoubleVector_nth_raw(self, i);
        if (PyErr_Occurred()) {
            Py_DECREF(result);
            return NULL;
        }

        // Add comma separator if not first element
        if (i > 0) {
            PyObject *comma = PyUnicode_FromString(", ");
            if (!comma) {
                Py_DECREF(result);
                return NULL;
            }
            PyObject *temp = PyUnicode_Concat(result, comma);
            Py_DECREF(comma);
            Py_DECREF(result);
            if (!temp) return NULL;
            result = temp;
        }

        // Convert double to Python float and get its repr
        PyObject *float_obj = PyFloat_FromDouble(val);
        if (!float_obj) {
            Py_DECREF(result);
            return NULL;
        }
        PyObject *val_str = PyObject_Repr(float_obj);
        Py_DECREF(float_obj);
        if (!val_str) {
            Py_DECREF(result);
            return NULL;
        }

        PyObject *new_result = PyUnicode_Concat(result, val_str);
        Py_DECREF(result);
        Py_DECREF(val_str);
        if (!new_result) return NULL;
        result = new_result;
    }

    PyObject *suffix = PyUnicode_FromString("])");
    if (!suffix) {
        Py_DECREF(result);
        return NULL;
    }

    PyObject *final = PyUnicode_Concat(result, suffix);
    Py_DECREF(result);
    Py_DECREF(suffix);
    return final;
}

static Py_hash_t DoubleVector_hash(DoubleVector *self) {
    if (self->hash_computed) {
        return self->hash;
    }

    Py_hash_t h = 0;
    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        double val = DoubleVector_nth_raw(self, i);
        // Hash the double bytes
        Py_hash_t item_hash = _Py_HashDouble((PyObject *)self, val);
        if (item_hash == -1 && PyErr_Occurred()) {
            return -1;
        }
        h = 31 * h + item_hash;
    }

    if (h == -1) h = -2;
    self->hash = h;
    self->hash_computed = 1;
    return h;
}

// Buffer Protocol Implementation for DoubleVector
static int DoubleVector_flatten(DoubleVector *self) {
    // Optimistic check - use atomic load for thread safety in free-threaded Python
#if HAVE_STDATOMIC
    if (atomic_load_explicit((_Atomic(double *)*)&self->flat_buffer_cache, memory_order_acquire) != NULL) {
        return 0;  // Already flattened
    }
#else
    if (self->flat_buffer_cache != NULL) {
        return 0;  // Already flattened
    }
#endif

    if (self->cnt == 0) {
        return 0;  // Empty, no buffer needed
    }

    double *buffer = (double *)malloc(self->cnt * sizeof(double));
    if (!buffer) {
        PyErr_NoMemory();
        return -1;
    }

    // Traverse trie and copy elements
    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        buffer[i] = DoubleVector_nth_raw(self, i);
        if (PyErr_Occurred()) {
            free(buffer);
            return -1;
        }
    }

    // Atomic CAS: Try to swap NULL with our new buffer
    // If another thread beat us, free ours and use theirs
#if HAVE_STDATOMIC
    double *expected = NULL;
    if (!atomic_compare_exchange_strong_explicit(
            (_Atomic(double *)*)&self->flat_buffer_cache,
            &expected,
            buffer,
            memory_order_release,
            memory_order_acquire)) {
        // We lost the race; another thread set it. Free ours.
        free(buffer);
    }
#else
    self->flat_buffer_cache = buffer;
#endif
    return 0;
}

static int DoubleVector_getbuffer(DoubleVector *self, Py_buffer *view, int flags) {
    if (self->cnt == 0) {
        // Empty vector - provide empty buffer
        view->buf = NULL;
        view->obj = (PyObject *)self;
        Py_INCREF(self);
        view->len = 0;
        view->readonly = 1;
        view->itemsize = sizeof(double);
        view->format = "d";
        view->ndim = 1;
        view->shape = NULL;
        view->strides = NULL;
        view->suboffsets = NULL;
        view->internal = NULL;
        return 0;
    }

    if (DoubleVector_flatten(self) < 0) {
        return -1;
    }

    view->buf = self->flat_buffer_cache;
    view->obj = (PyObject *)self;
    Py_INCREF(self);
    view->len = self->cnt * sizeof(double);
    view->readonly = 1;  // Immutable!
    view->itemsize = sizeof(double);
    view->format = "d";
    view->ndim = 1;
    view->shape = &self->cnt;
    view->strides = NULL;
    view->suboffsets = NULL;
    view->internal = NULL;

    return 0;
}

static void DoubleVector_releasebuffer(DoubleVector *self, Py_buffer *view) {
    // No-op: keep cache for object lifetime
}

static PyBufferProcs DoubleVector_as_buffer = {
    .bf_getbuffer = (getbufferproc)DoubleVector_getbuffer,
    .bf_releasebuffer = (releasebufferproc)DoubleVector_releasebuffer,
};

// DoubleVector iterator
typedef struct {
    PyObject_HEAD
    DoubleVector *vec;
    Py_ssize_t index;
} DoubleVectorIterator;

static PyTypeObject DoubleVectorIteratorType;

static void DoubleVectorIterator_dealloc(DoubleVectorIterator *self) {
    Py_XDECREF(self->vec);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DoubleVectorIterator_next(DoubleVectorIterator *self) {
    if (self->index >= self->vec->cnt) {
        return NULL;  // StopIteration
    }

    double val = DoubleVector_nth_raw(self->vec, self->index);
    if (PyErr_Occurred()) return NULL;
    self->index++;
    return PyFloat_FromDouble(val);
}

static PyTypeObject DoubleVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.DoubleVectorIterator",
    .tp_basicsize = sizeof(DoubleVectorIterator),
    .tp_dealloc = (destructor)DoubleVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)DoubleVectorIterator_next,
};

static PyObject *DoubleVector_iter(DoubleVector *self) {
    DoubleVectorIterator *it = PyObject_New(DoubleVectorIterator, &DoubleVectorIteratorType);
    if (!it) return NULL;
    it->vec = self;
    Py_INCREF(self);
    it->index = 0;
    return (PyObject *)it;
}

// === TransientDoubleVector ===
typedef struct TransientDoubleVector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    DoubleVectorNode *root;
    double *tail;
    Py_ssize_t tail_len;
    Py_ssize_t tail_cap;
    PyObject *id;
} TransientDoubleVector;

static PyTypeObject TransientDoubleVectorType;

static void TransientDoubleVector_dealloc(TransientDoubleVector *self) {
    Py_XDECREF(self->root);
    if (self->tail) free(self->tail);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void TransientDoubleVector_ensure_editable(TransientDoubleVector *self) {
    if (self->id == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Transient used after persistent() call");
    }
}

static Py_ssize_t TransientDoubleVector_tail_off(TransientDoubleVector *self) {
    if (self->cnt < WIDTH) {
        return 0;
    }
    return ((self->cnt - 1) >> BITS) << BITS;
}

static DoubleVectorNode *TransientDoubleVector_new_path(TransientDoubleVector *self, int level, DoubleVectorNode *node) {
    if (level == 0) {
        Py_INCREF(node);
        return node;
    }
    DoubleVectorNode *ret = DoubleVectorNode_create(self->id);
    if (!ret) return NULL;

    DoubleVectorNode *child = TransientDoubleVector_new_path(self, level - BITS, node);
    if (!child) {
        Py_DECREF(ret);
        return NULL;
    }
    ret->data.children[0] = child;
    ret->valid_mask = 1;
    return ret;
}

static DoubleVectorNode *TransientDoubleVector_push_tail(TransientDoubleVector *self, int level, DoubleVectorNode *parent, DoubleVectorNode *tail_node) {
    int subidx = ((self->cnt - 1) >> level) & MASK;
    DoubleVectorNode *ret;

    if (DoubleVectorNode_is_editable(parent, self->id)) {
        ret = parent;
        Py_INCREF(ret);
    } else {
        ret = DoubleVectorNode_clone(parent, self->id);
        if (!ret) return NULL;
    }

    DoubleVectorNode *node_to_insert;
    if (level == BITS) {
        node_to_insert = tail_node;
        Py_INCREF(tail_node);
    } else {
        DoubleVectorNode *child = parent->data.children[subidx];
        if (child != NULL && (parent->valid_mask & (1 << subidx))) {
            node_to_insert = TransientDoubleVector_push_tail(self, level - BITS, child, tail_node);
        } else {
            node_to_insert = TransientDoubleVector_new_path(self, level - BITS, tail_node);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    if (ret->valid_mask & (1 << subidx)) {
        Py_XDECREF(ret->data.children[subidx]);
    }
    ret->data.children[subidx] = node_to_insert;
    ret->valid_mask |= (1 << subidx);
    return ret;
}

// Internal function that takes a raw double value directly (no boxing overhead)
static int TransientDoubleVector_conj_mut_raw(TransientDoubleVector *self, double dval) {
    // Room in tail?
    if (self->cnt - TransientDoubleVector_tail_off(self) < WIDTH) {
        // Grow tail if needed
        if (self->tail_len >= self->tail_cap) {
            Py_ssize_t new_cap = self->tail_cap == 0 ? WIDTH : self->tail_cap * 2;
            if (new_cap > WIDTH) new_cap = WIDTH;
            double *new_tail = (double *)realloc(self->tail, new_cap * sizeof(double));
            if (!new_tail) {
                PyErr_NoMemory();
                return -1;
            }
            self->tail = new_tail;
            self->tail_cap = new_cap;
        }
        self->tail[self->tail_len++] = dval;
        self->cnt++;
        return 0;
    }

    // Tail is full, push into trie
    DoubleVectorNode *tail_node = DoubleVectorNode_create(self->id);
    if (!tail_node) return -1;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1 << i);
    }

    // Reset tail
    self->tail[0] = dval;
    self->tail_len = 1;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        DoubleVectorNode *new_root = DoubleVectorNode_create(self->id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return -1;
        }
        new_root->data.children[0] = self->root;
        Py_INCREF(self->root);
        new_root->valid_mask = 1;

        DoubleVectorNode *path = TransientDoubleVector_new_path(self, self->shift, tail_node);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return -1;
        }
        new_root->data.children[1] = path;
        new_root->valid_mask |= 2;

        Py_DECREF(self->root);
        self->root = new_root;
        self->shift += BITS;
    } else {
        DoubleVectorNode *new_root = TransientDoubleVector_push_tail(self, self->shift, self->root, tail_node);
        if (!new_root) {
            Py_DECREF(tail_node);
            return -1;
        }
        Py_DECREF(self->root);
        self->root = new_root;
    }

    Py_DECREF(tail_node);
    self->cnt++;

    return 0;
}

static PyObject *TransientDoubleVector_conj_mut(TransientDoubleVector *self, PyObject *val) {
    TransientDoubleVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    // Unbox the value
    double dval = PyFloat_AsDouble(val);
    if (dval == -1.0 && PyErr_Occurred()) {
        return NULL;
    }

    // Room in tail?
    if (self->cnt - TransientDoubleVector_tail_off(self) < WIDTH) {
        // Grow tail if needed
        if (self->tail_len >= self->tail_cap) {
            Py_ssize_t new_cap = self->tail_cap == 0 ? WIDTH : self->tail_cap * 2;
            if (new_cap > WIDTH) new_cap = WIDTH;
            double *new_tail = (double *)realloc(self->tail, new_cap * sizeof(double));
            if (!new_tail) {
                PyErr_NoMemory();
                return NULL;
            }
            self->tail = new_tail;
            self->tail_cap = new_cap;
        }
        self->tail[self->tail_len++] = dval;
        self->cnt++;
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Tail is full, push into trie
    DoubleVectorNode *tail_node = DoubleVectorNode_create(self->id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1 << i);
    }

    // Reset tail
    self->tail[0] = dval;
    self->tail_len = 1;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        DoubleVectorNode *new_root = DoubleVectorNode_create(self->id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[0] = self->root;
        Py_INCREF(self->root);
        new_root->valid_mask = 1;

        DoubleVectorNode *path = TransientDoubleVector_new_path(self, self->shift, tail_node);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[1] = path;
        new_root->valid_mask |= 2;

        Py_DECREF(self->root);
        self->root = new_root;
        self->shift += BITS;
    } else {
        DoubleVectorNode *new_root = TransientDoubleVector_push_tail(self, self->shift, self->root, tail_node);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        Py_DECREF(self->root);
        self->root = new_root;
    }

    Py_DECREF(tail_node);
    self->cnt++;

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientDoubleVector_persistent(TransientDoubleVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientDoubleVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_CLEAR(self->id);

    DoubleVector *result = DoubleVector_create(self->cnt, self->shift, self->root,
                                                  self->tail, self->tail_len, NULL);
    return (PyObject *)result;
}

static PyMethodDef TransientDoubleVector_methods[] = {
    {"conj_mut", (PyCFunction)TransientDoubleVector_conj_mut, METH_O, "Mutably add element"},
    {"persistent", (PyCFunction)TransientDoubleVector_persistent, METH_NOARGS, "Return persistent vector"},
    {NULL}
};

static PyTypeObject TransientDoubleVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientDoubleVector",
    .tp_doc = "Transient double vector for batch operations",
    .tp_basicsize = sizeof(TransientDoubleVector),
    .tp_dealloc = (destructor)TransientDoubleVector_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = TransientDoubleVector_methods,
};

static PyObject *DoubleVector_transient(DoubleVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientDoubleVector *t = PyObject_New(TransientDoubleVector, &TransientDoubleVectorType);
    if (!t) return NULL;

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        return NULL;
    }

    t->cnt = self->cnt;
    t->shift = self->shift;
    t->root = DoubleVectorNode_clone(self->root, t->id);
    if (!t->root) {
        Py_DECREF(t);
        return NULL;
    }

    if (self->tail && self->tail_len > 0) {
        t->tail = (double *)malloc(WIDTH * sizeof(double));
        if (!t->tail) {
            Py_DECREF(t);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(t->tail, self->tail, self->tail_len * sizeof(double));
        t->tail_len = self->tail_len;
        t->tail_cap = WIDTH;
    } else {
        t->tail = (double *)malloc(WIDTH * sizeof(double));
        if (!t->tail) {
            Py_DECREF(t);
            PyErr_NoMemory();
            return NULL;
        }
        t->tail_len = 0;
        t->tail_cap = WIDTH;
    }

    return (PyObject *)t;
}

static int DoubleVector_init(DoubleVector *self, PyObject *args, PyObject *kwds) {
    if (kwds && PyDict_GET_SIZE(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "DoubleVector takes no keyword arguments");
        return -1;
    }

    Py_ssize_t n = PyTuple_Size(args);

    if (n == 0) {
        return 0;  // Empty vector already set up in __new__
    }

    // Check if single argument that's an iterable (but not a string)
    if (n == 1) {
        PyObject *arg = PyTuple_GET_ITEM(args, 0);
        if (!PyUnicode_Check(arg) && !PyBytes_Check(arg)) {
            PyObject *iter = PyObject_GetIter(arg);
            if (iter != NULL) {
                PyObject *item;
                while ((item = PyIter_Next(iter)) != NULL) {
                    PyObject *new_vec = DoubleVector_conj(self, item);
                    Py_DECREF(item);
                    if (!new_vec) {
                        Py_DECREF(iter);
                        return -1;
                    }

                    // Update self from new_vec
                    DoubleVector *nv = (DoubleVector *)new_vec;
                    Py_DECREF(self->root);
                    if (self->tail) free(self->tail);
                    self->cnt = nv->cnt;
                    self->shift = nv->shift;
                    self->root = nv->root;
                    Py_INCREF(self->root);
                    if (nv->tail && nv->tail_len > 0) {
                        self->tail = (double *)malloc(nv->tail_len * sizeof(double));
                        memcpy(self->tail, nv->tail, nv->tail_len * sizeof(double));
                        self->tail_len = nv->tail_len;
                        self->tail_cap = nv->tail_len;
                    } else {
                        self->tail = NULL;
                        self->tail_len = 0;
                        self->tail_cap = 0;
                    }
                    Py_DECREF(new_vec);
                }
                Py_DECREF(iter);

                if (PyErr_Occurred()) return -1;
                return 0;
            }
            PyErr_Clear();
        }
    }

    // Multiple arguments or single non-iterable: treat as varargs
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        PyObject *new_vec = DoubleVector_conj(self, item);
        if (!new_vec) {
            return -1;
        }

        // Update self from new_vec
        DoubleVector *nv = (DoubleVector *)new_vec;
        Py_DECREF(self->root);
        if (self->tail) free(self->tail);
        self->cnt = nv->cnt;
        self->shift = nv->shift;
        self->root = nv->root;
        Py_INCREF(self->root);
        if (nv->tail && nv->tail_len > 0) {
            self->tail = (double *)malloc(nv->tail_len * sizeof(double));
            memcpy(self->tail, nv->tail, nv->tail_len * sizeof(double));
            self->tail_len = nv->tail_len;
            self->tail_cap = nv->tail_len;
        } else {
            self->tail = NULL;
            self->tail_len = 0;
            self->tail_cap = 0;
        }
        Py_DECREF(new_vec);
    }

    return 0;
}

static PyObject *DoubleVector_reduce(DoubleVector *self, PyObject *Py_UNUSED(ignored)) {
    // Convert DoubleVector to a tuple using the sequence protocol
    PyObject *args = PySequence_Tuple((PyObject *)self);
    if (args == NULL) {
        return NULL;
    }

    // Return (type, args_tuple) - pickle will call type(*args_tuple)
    PyObject *result = PyTuple_Pack(2, (PyObject *)Py_TYPE(self), args);
    Py_DECREF(args);
    return result;
}

static PyMethodDef DoubleVector_methods[] = {
    {"nth", (PyCFunction)DoubleVector_nth, METH_VARARGS, "Get element at index"},
    {"conj", (PyCFunction)DoubleVector_conj, METH_O, "Add element to end"},
    {"transient", (PyCFunction)DoubleVector_transient, METH_NOARGS, "Return transient version for batch operations"},
    {"__reduce__", (PyCFunction)DoubleVector_reduce, METH_NOARGS, "Pickle support"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations"},
    {NULL}
};

static PyMappingMethods DoubleVector_as_mapping = {
    .mp_length = (lenfunc)DoubleVector_length,
    .mp_subscript = (binaryfunc)DoubleVector_getitem,
};

// Sequence protocol sq_item - takes Py_ssize_t index directly
static PyObject *DoubleVector_sq_item(DoubleVector *self, Py_ssize_t i) {
    if (i < 0) {
        i = self->cnt + i;
    }
    if (i < 0 || i >= self->cnt) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    double val = DoubleVector_nth_raw(self, i);
    if (PyErr_Occurred()) return NULL;
    return PyFloat_FromDouble(val);
}

static PySequenceMethods DoubleVector_as_sequence = {
    .sq_length = (lenfunc)DoubleVector_length,
    .sq_item = (ssizeargfunc)DoubleVector_sq_item,
};

static PyTypeObject DoubleVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.DoubleVector",
    .tp_doc = "Persistent Vector of doubles (float64) with buffer protocol support",
    .tp_basicsize = sizeof(DoubleVector),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)DoubleVector_dealloc,
    .tp_repr = (reprfunc)DoubleVector_repr,
    .tp_as_buffer = &DoubleVector_as_buffer,
    .tp_as_sequence = &DoubleVector_as_sequence,
    .tp_as_mapping = &DoubleVector_as_mapping,
    .tp_hash = (hashfunc)DoubleVector_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = (getiterfunc)DoubleVector_iter,
    .tp_methods = DoubleVector_methods,
    .tp_init = (initproc)DoubleVector_init,
    .tp_new = DoubleVector_new,
};

// =============================================================================
// INTVECTOR - Type-specialized vector for integers (int64)
// =============================================================================

// Forward declarations for IntVector
static PyObject *IntVector_conj(IntVector *self, PyObject *val);

// --- IntVectorNode ---
typedef struct IntVectorNode {
    PyObject_HEAD
    union {
        int64_t values[WIDTH];
        struct IntVectorNode *children[WIDTH];
    } data;
    int valid_mask;
    PyObject *transient_id;
} IntVectorNode;

static PyTypeObject IntVectorNodeType;

static void IntVectorNode_dealloc(IntVectorNode *self) {
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static IntVectorNode *IntVectorNode_create(PyObject *transient_id) {
    IntVectorNode *node = PyObject_New(IntVectorNode, &IntVectorNodeType);
    if (!node) return NULL;

    for (int i = 0; i < WIDTH; i++) {
        node->data.values[i] = 0;
        node->data.children[i] = NULL;
    }
    node->valid_mask = 0;
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);
    return node;
}

static IntVectorNode *IntVectorNode_clone(IntVectorNode *self, PyObject *transient_id) {
    IntVectorNode *node = IntVectorNode_create(transient_id);
    if (!node) return NULL;

    memcpy(&node->data, &self->data, sizeof(self->data));
    node->valid_mask = self->valid_mask;
    return node;
}

static int IntVectorNode_is_editable(IntVectorNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

static PyTypeObject IntVectorNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.IntVectorNode",
    .tp_basicsize = sizeof(IntVectorNode),
    .tp_dealloc = (destructor)IntVectorNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// Global empty long node
static IntVectorNode *EMPTY_LONG_NODE = NULL;

// --- IntVector ---
typedef struct IntVector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    IntVectorNode *root;
    int64_t *tail;
    Py_ssize_t tail_len;
    Py_ssize_t tail_cap;
    Py_hash_t hash;
    int hash_computed;
    PyObject *transient_id;
    // Buffer protocol cache
    int64_t *flat_buffer_cache;
} IntVector;

static PyTypeObject IntVectorType;
static IntVector *EMPTY_LONG_VECTOR = NULL;

static void IntVector_dealloc(IntVector *self) {
    Py_XDECREF(self->root);
    if (self->tail) free(self->tail);
    if (self->flat_buffer_cache) free(self->flat_buffer_cache);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *IntVector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    IntVector *self = (IntVector *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->cnt = 0;
        self->shift = BITS;
        self->root = EMPTY_LONG_NODE;
        Py_INCREF(EMPTY_LONG_NODE);
        self->tail = NULL;
        self->tail_len = 0;
        self->tail_cap = 0;
        self->hash = 0;
        self->hash_computed = 0;
        self->transient_id = NULL;
        self->flat_buffer_cache = NULL;
    }
    return (PyObject *)self;
}

static IntVector *IntVector_create(Py_ssize_t cnt, int shift, IntVectorNode *root,
                                       int64_t *tail, Py_ssize_t tail_len, PyObject *transient_id) {
    IntVector *vec = (IntVector *)IntVectorType.tp_alloc(&IntVectorType, 0);
    if (!vec) return NULL;

    vec->cnt = cnt;
    vec->shift = shift;
    vec->root = root ? root : EMPTY_LONG_NODE;
    Py_INCREF(vec->root);

    if (tail && tail_len > 0) {
        vec->tail = (int64_t *)malloc(tail_len * sizeof(int64_t));
        if (!vec->tail) {
            Py_DECREF(vec);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(vec->tail, tail, tail_len * sizeof(int64_t));
        vec->tail_len = tail_len;
        vec->tail_cap = tail_len;
    } else {
        vec->tail = NULL;
        vec->tail_len = 0;
        vec->tail_cap = 0;
    }

    vec->hash = 0;
    vec->hash_computed = 0;
    vec->transient_id = transient_id;
    Py_XINCREF(transient_id);
    vec->flat_buffer_cache = NULL;

    return vec;
}

static Py_ssize_t IntVector_length(IntVector *self) {
    return self->cnt;
}

static Py_ssize_t IntVector_tail_off(IntVector *self) {
    if (self->cnt < WIDTH) {
        return 0;
    }
    return ((self->cnt - 1) >> BITS) << BITS;
}

static int64_t *IntVector_array_for(IntVector *self, Py_ssize_t i) {
    if (i < 0 || i >= self->cnt) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    if (i >= IntVector_tail_off(self)) {
        return self->tail;
    }

    IntVectorNode *node = self->root;
    for (int level = self->shift; level > 0; level -= BITS) {
        int idx = (i >> level) & MASK;
        node = node->data.children[idx];
    }
    return (int64_t *)node->data.values;
}

static int64_t IntVector_nth_raw(IntVector *self, Py_ssize_t i) {
    int64_t *arr = IntVector_array_for(self, i);
    if (!arr) return 0;
    return arr[i & MASK];
}

static PyObject *IntVector_nth(IntVector *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *default_val = NULL;

    if (!PyArg_ParseTuple(args, "n|O", &i, &default_val)) {
        return NULL;
    }

    if (i < 0) {
        i = self->cnt + i;
    }

    if (i < 0 || i >= self->cnt) {
        if (default_val) {
            Py_INCREF(default_val);
            return default_val;
        }
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int64_t val = IntVector_nth_raw(self, i);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLongLong(val);
}

static PyObject *IntVector_getitem(IntVector *self, PyObject *key) {
    if (PyIndex_Check(key)) {
        Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred()) return NULL;

        if (i < 0) i = self->cnt + i;
        if (i < 0 || i >= self->cnt) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        int64_t val = IntVector_nth_raw(self, i);
        if (PyErr_Occurred()) return NULL;
        return PyLong_FromLongLong(val);
    }

    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelength;
        if (PySlice_GetIndicesEx(key, self->cnt, &start, &stop, &step, &slicelength) < 0) {
            return NULL;
        }

        IntVector *result = IntVector_create(0, BITS, NULL, NULL, 0, NULL);
        if (!result) return NULL;

        for (Py_ssize_t i = start, j = 0; j < slicelength; i += step, j++) {
            int64_t val = IntVector_nth_raw(self, i);
            if (PyErr_Occurred()) {
                Py_DECREF(result);
                return NULL;
            }
            PyObject *boxed = PyLong_FromLongLong(val);
            if (!boxed) {
                Py_DECREF(result);
                return NULL;
            }
            PyObject *new_result = IntVector_conj(result, boxed);
            Py_DECREF(boxed);
            Py_DECREF(result);
            if (!new_result) return NULL;
            result = (IntVector *)new_result;
        }

        return (PyObject *)result;
    }

    PyErr_SetString(PyExc_TypeError, "indices must be integers or slices");
    return NULL;
}

static IntVectorNode *IntVector_new_path(IntVector *self, int level, IntVectorNode *node, PyObject *transient_id) {
    if (level == 0) {
        Py_INCREF(node);
        return node;
    }
    IntVectorNode *ret = IntVectorNode_create(transient_id);
    if (!ret) return NULL;

    IntVectorNode *child = IntVector_new_path(self, level - BITS, node, transient_id);
    if (!child) {
        Py_DECREF(ret);
        return NULL;
    }
    ret->data.children[0] = child;
    ret->valid_mask = 1;
    return ret;
}

static IntVectorNode *IntVector_push_tail(IntVector *self, int level, IntVectorNode *parent, IntVectorNode *tail_node, PyObject *transient_id) {
    int subidx = ((self->cnt - 1) >> level) & MASK;
    IntVectorNode *ret;

    if (IntVectorNode_is_editable(parent, transient_id)) {
        ret = parent;
        Py_INCREF(ret);
    } else {
        ret = IntVectorNode_clone(parent, transient_id);
        if (!ret) return NULL;
    }

    IntVectorNode *node_to_insert;
    if (level == BITS) {
        node_to_insert = tail_node;
        Py_INCREF(tail_node);
    } else {
        IntVectorNode *child = parent->data.children[subidx];
        if (child != NULL && (parent->valid_mask & (1 << subidx))) {
            node_to_insert = IntVector_push_tail(self, level - BITS, child, tail_node, transient_id);
        } else {
            node_to_insert = IntVector_new_path(self, level - BITS, tail_node, transient_id);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    if (ret->valid_mask & (1 << subidx)) {
        Py_XDECREF(ret->data.children[subidx]);
    }
    ret->data.children[subidx] = node_to_insert;
    ret->valid_mask |= (1 << subidx);
    return ret;
}

static PyObject *IntVector_conj(IntVector *self, PyObject *val) {
    int64_t lval = PyLong_AsLongLong(val);
    if (lval == -1 && PyErr_Occurred()) {
        return NULL;
    }

    PyObject *transient_id = self->transient_id;

    // Room in tail?
    if (self->cnt - IntVector_tail_off(self) < WIDTH) {
        Py_ssize_t new_tail_len = self->tail_len + 1;
        int64_t *new_tail = (int64_t *)malloc(new_tail_len * sizeof(int64_t));
        if (!new_tail) {
            PyErr_NoMemory();
            return NULL;
        }

        if (self->tail && self->tail_len > 0) {
            memcpy(new_tail, self->tail, self->tail_len * sizeof(int64_t));
        }
        new_tail[self->tail_len] = lval;

        IntVector *result = IntVector_create(self->cnt + 1, self->shift, self->root,
                                                  new_tail, new_tail_len, transient_id);
        free(new_tail);
        return (PyObject *)result;
    }

    // Tail is full, push into trie
    IntVectorNode *tail_node = IntVectorNode_create(transient_id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1 << i);
    }

    int new_shift = self->shift;
    IntVectorNode *new_root;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        new_root = IntVectorNode_create(transient_id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[0] = self->root;
        Py_INCREF(self->root);
        new_root->valid_mask = 1;

        IntVectorNode *path = IntVector_new_path(self, self->shift, tail_node, transient_id);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[1] = path;
        new_root->valid_mask |= 2;
        new_shift += BITS;
    } else {
        new_root = IntVector_push_tail(self, self->shift, self->root, tail_node, transient_id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
    }

    Py_DECREF(tail_node);

    int64_t new_tail_arr[1] = { lval };
    IntVector *result = IntVector_create(self->cnt + 1, new_shift, new_root,
                                              new_tail_arr, 1, transient_id);
    Py_DECREF(new_root);
    return (PyObject *)result;
}

static PyObject *IntVector_repr(IntVector *self) {
    PyObject *result = PyUnicode_FromString("vec_i64([");
    if (!result) return NULL;

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        int64_t val = IntVector_nth_raw(self, i);
        if (PyErr_Occurred()) {
            Py_DECREF(result);
            return NULL;
        }

        // Add comma separator if not first element
        if (i > 0) {
            PyObject *comma = PyUnicode_FromString(", ");
            if (!comma) {
                Py_DECREF(result);
                return NULL;
            }
            PyObject *temp = PyUnicode_Concat(result, comma);
            Py_DECREF(comma);
            Py_DECREF(result);
            if (!temp) return NULL;
            result = temp;
        }

        // Convert int64 to Python int and get its repr
        PyObject *long_obj = PyLong_FromLongLong(val);
        if (!long_obj) {
            Py_DECREF(result);
            return NULL;
        }
        PyObject *val_str = PyObject_Repr(long_obj);
        Py_DECREF(long_obj);
        if (!val_str) {
            Py_DECREF(result);
            return NULL;
        }

        PyObject *new_result = PyUnicode_Concat(result, val_str);
        Py_DECREF(result);
        Py_DECREF(val_str);
        if (!new_result) return NULL;
        result = new_result;
    }

    PyObject *suffix = PyUnicode_FromString("])");
    if (!suffix) {
        Py_DECREF(result);
        return NULL;
    }

    PyObject *final = PyUnicode_Concat(result, suffix);
    Py_DECREF(result);
    Py_DECREF(suffix);
    return final;
}

static Py_hash_t IntVector_hash(IntVector *self) {
    if (self->hash_computed) {
        return self->hash;
    }

    Py_hash_t h = 0;
    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        int64_t val = IntVector_nth_raw(self, i);
        Py_hash_t item_hash = (Py_hash_t)val;
        if (item_hash == -1) item_hash = -2;
        h = 31 * h + item_hash;
    }

    if (h == -1) h = -2;
    self->hash = h;
    self->hash_computed = 1;
    return h;
}

// Buffer Protocol for IntVector
static int IntVector_flatten(IntVector *self) {
    // Optimistic check - use atomic load for thread safety in free-threaded Python
#if HAVE_STDATOMIC
    if (atomic_load_explicit((_Atomic(int64_t *)*)&self->flat_buffer_cache, memory_order_acquire) != NULL) {
        return 0;
    }
#else
    if (self->flat_buffer_cache != NULL) {
        return 0;
    }
#endif

    if (self->cnt == 0) {
        return 0;
    }

    int64_t *buffer = (int64_t *)malloc(self->cnt * sizeof(int64_t));
    if (!buffer) {
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        buffer[i] = IntVector_nth_raw(self, i);
        if (PyErr_Occurred()) {
            free(buffer);
            return -1;
        }
    }

    // Atomic CAS: Try to swap NULL with our new buffer
    // If another thread beat us, free ours and use theirs
#if HAVE_STDATOMIC
    int64_t *expected = NULL;
    if (!atomic_compare_exchange_strong_explicit(
            (_Atomic(int64_t *)*)&self->flat_buffer_cache,
            &expected,
            buffer,
            memory_order_release,
            memory_order_acquire)) {
        // We lost the race; another thread set it. Free ours.
        free(buffer);
    }
#else
    self->flat_buffer_cache = buffer;
#endif
    return 0;
}

static int IntVector_getbuffer(IntVector *self, Py_buffer *view, int flags) {
    if (self->cnt == 0) {
        view->buf = NULL;
        view->obj = (PyObject *)self;
        Py_INCREF(self);
        view->len = 0;
        view->readonly = 1;
        view->itemsize = sizeof(int64_t);
        view->format = "q";
        view->ndim = 1;
        view->shape = NULL;
        view->strides = NULL;
        view->suboffsets = NULL;
        view->internal = NULL;
        return 0;
    }

    if (IntVector_flatten(self) < 0) {
        return -1;
    }

    view->buf = self->flat_buffer_cache;
    view->obj = (PyObject *)self;
    Py_INCREF(self);
    view->len = self->cnt * sizeof(int64_t);
    view->readonly = 1;
    view->itemsize = sizeof(int64_t);
    view->format = "q";
    view->ndim = 1;
    view->shape = &self->cnt;
    view->strides = NULL;
    view->suboffsets = NULL;
    view->internal = NULL;

    return 0;
}

static void IntVector_releasebuffer(IntVector *self, Py_buffer *view) {
    // No-op
}

static PyBufferProcs IntVector_as_buffer = {
    .bf_getbuffer = (getbufferproc)IntVector_getbuffer,
    .bf_releasebuffer = (releasebufferproc)IntVector_releasebuffer,
};

// IntVector iterator
typedef struct {
    PyObject_HEAD
    IntVector *vec;
    Py_ssize_t index;
} IntVectorIterator;

static PyTypeObject IntVectorIteratorType;

static void IntVectorIterator_dealloc(IntVectorIterator *self) {
    Py_XDECREF(self->vec);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *IntVectorIterator_next(IntVectorIterator *self) {
    if (self->index >= self->vec->cnt) {
        return NULL;
    }

    int64_t val = IntVector_nth_raw(self->vec, self->index);
    if (PyErr_Occurred()) return NULL;
    self->index++;
    return PyLong_FromLongLong(val);
}

static PyTypeObject IntVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.IntVectorIterator",
    .tp_basicsize = sizeof(IntVectorIterator),
    .tp_dealloc = (destructor)IntVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)IntVectorIterator_next,
};

static PyObject *IntVector_iter(IntVector *self) {
    IntVectorIterator *it = PyObject_New(IntVectorIterator, &IntVectorIteratorType);
    if (!it) return NULL;
    it->vec = self;
    Py_INCREF(self);
    it->index = 0;
    return (PyObject *)it;
}

// === TransientIntVector ===
typedef struct TransientIntVector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    IntVectorNode *root;
    int64_t *tail;
    Py_ssize_t tail_len;
    Py_ssize_t tail_cap;
    PyObject *id;
} TransientIntVector;

static PyTypeObject TransientIntVectorType;

static void TransientIntVector_dealloc(TransientIntVector *self) {
    Py_XDECREF(self->root);
    if (self->tail) free(self->tail);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void TransientIntVector_ensure_editable(TransientIntVector *self) {
    if (self->id == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Transient used after persistent() call");
    }
}

static Py_ssize_t TransientIntVector_tail_off(TransientIntVector *self) {
    if (self->cnt < WIDTH) {
        return 0;
    }
    return ((self->cnt - 1) >> BITS) << BITS;
}

static IntVectorNode *TransientIntVector_new_path(TransientIntVector *self, int level, IntVectorNode *node) {
    if (level == 0) {
        Py_INCREF(node);
        return node;
    }
    IntVectorNode *ret = IntVectorNode_create(self->id);
    if (!ret) return NULL;

    IntVectorNode *child = TransientIntVector_new_path(self, level - BITS, node);
    if (!child) {
        Py_DECREF(ret);
        return NULL;
    }
    ret->data.children[0] = child;
    ret->valid_mask = 1;
    return ret;
}

static IntVectorNode *TransientIntVector_push_tail(TransientIntVector *self, int level, IntVectorNode *parent, IntVectorNode *tail_node) {
    int subidx = ((self->cnt - 1) >> level) & MASK;
    IntVectorNode *ret;

    if (IntVectorNode_is_editable(parent, self->id)) {
        ret = parent;
        Py_INCREF(ret);
    } else {
        ret = IntVectorNode_clone(parent, self->id);
        if (!ret) return NULL;
    }

    IntVectorNode *node_to_insert;
    if (level == BITS) {
        node_to_insert = tail_node;
        Py_INCREF(tail_node);
    } else {
        IntVectorNode *child = parent->data.children[subidx];
        if (child != NULL && (parent->valid_mask & (1 << subidx))) {
            node_to_insert = TransientIntVector_push_tail(self, level - BITS, child, tail_node);
        } else {
            node_to_insert = TransientIntVector_new_path(self, level - BITS, tail_node);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    if (ret->valid_mask & (1 << subidx)) {
        Py_XDECREF(ret->data.children[subidx]);
    }
    ret->data.children[subidx] = node_to_insert;
    ret->valid_mask |= (1 << subidx);
    return ret;
}

// Internal function that takes a raw int64_t value directly (no boxing overhead)
static int TransientIntVector_conj_mut_raw(TransientIntVector *self, int64_t lval) {
    // Room in tail?
    if (self->cnt - TransientIntVector_tail_off(self) < WIDTH) {
        // Grow tail if needed
        if (self->tail_len >= self->tail_cap) {
            Py_ssize_t new_cap = self->tail_cap == 0 ? WIDTH : self->tail_cap * 2;
            if (new_cap > WIDTH) new_cap = WIDTH;
            int64_t *new_tail = (int64_t *)realloc(self->tail, new_cap * sizeof(int64_t));
            if (!new_tail) {
                PyErr_NoMemory();
                return -1;
            }
            self->tail = new_tail;
            self->tail_cap = new_cap;
        }
        self->tail[self->tail_len++] = lval;
        self->cnt++;
        return 0;
    }

    // Tail is full, push into trie
    IntVectorNode *tail_node = IntVectorNode_create(self->id);
    if (!tail_node) return -1;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1 << i);
    }

    // Reset tail
    self->tail[0] = lval;
    self->tail_len = 1;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        IntVectorNode *new_root = IntVectorNode_create(self->id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return -1;
        }
        new_root->data.children[0] = self->root;
        Py_INCREF(self->root);
        new_root->valid_mask = 1;

        IntVectorNode *path = TransientIntVector_new_path(self, self->shift, tail_node);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return -1;
        }
        new_root->data.children[1] = path;
        new_root->valid_mask |= 2;

        Py_DECREF(self->root);
        self->root = new_root;
        self->shift += BITS;
    } else {
        IntVectorNode *new_root = TransientIntVector_push_tail(self, self->shift, self->root, tail_node);
        if (!new_root) {
            Py_DECREF(tail_node);
            return -1;
        }
        Py_DECREF(self->root);
        self->root = new_root;
    }

    Py_DECREF(tail_node);
    self->cnt++;

    return 0;
}

static PyObject *TransientIntVector_conj_mut(TransientIntVector *self, PyObject *val) {
    TransientIntVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    // Unbox the value
    int64_t lval = PyLong_AsLongLong(val);
    if (lval == -1 && PyErr_Occurred()) {
        return NULL;
    }

    // Room in tail?
    if (self->cnt - TransientIntVector_tail_off(self) < WIDTH) {
        // Grow tail if needed
        if (self->tail_len >= self->tail_cap) {
            Py_ssize_t new_cap = self->tail_cap == 0 ? WIDTH : self->tail_cap * 2;
            if (new_cap > WIDTH) new_cap = WIDTH;
            int64_t *new_tail = (int64_t *)realloc(self->tail, new_cap * sizeof(int64_t));
            if (!new_tail) {
                PyErr_NoMemory();
                return NULL;
            }
            self->tail = new_tail;
            self->tail_cap = new_cap;
        }
        self->tail[self->tail_len++] = lval;
        self->cnt++;
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Tail is full, push into trie
    IntVectorNode *tail_node = IntVectorNode_create(self->id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1 << i);
    }

    // Reset tail
    self->tail[0] = lval;
    self->tail_len = 1;

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        IntVectorNode *new_root = IntVectorNode_create(self->id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[0] = self->root;
        Py_INCREF(self->root);
        new_root->valid_mask = 1;

        IntVectorNode *path = TransientIntVector_new_path(self, self->shift, tail_node);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->data.children[1] = path;
        new_root->valid_mask |= 2;

        Py_DECREF(self->root);
        self->root = new_root;
        self->shift += BITS;
    } else {
        IntVectorNode *new_root = TransientIntVector_push_tail(self, self->shift, self->root, tail_node);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        Py_DECREF(self->root);
        self->root = new_root;
    }

    Py_DECREF(tail_node);
    self->cnt++;

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientIntVector_persistent(TransientIntVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientIntVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_CLEAR(self->id);

    IntVector *result = IntVector_create(self->cnt, self->shift, self->root,
                                              self->tail, self->tail_len, NULL);
    return (PyObject *)result;
}

static PyMethodDef TransientIntVector_methods[] = {
    {"conj_mut", (PyCFunction)TransientIntVector_conj_mut, METH_O, "Mutably add element"},
    {"persistent", (PyCFunction)TransientIntVector_persistent, METH_NOARGS, "Return persistent vector"},
    {NULL}
};

static PyTypeObject TransientIntVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientIntVector",
    .tp_doc = "Transient long vector for batch operations",
    .tp_basicsize = sizeof(TransientIntVector),
    .tp_dealloc = (destructor)TransientIntVector_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = TransientIntVector_methods,
};

static PyObject *IntVector_transient(IntVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientIntVector *t = PyObject_New(TransientIntVector, &TransientIntVectorType);
    if (!t) return NULL;

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        return NULL;
    }

    t->cnt = self->cnt;
    t->shift = self->shift;
    t->root = IntVectorNode_clone(self->root, t->id);
    if (!t->root) {
        Py_DECREF(t);
        return NULL;
    }

    if (self->tail && self->tail_len > 0) {
        t->tail = (int64_t *)malloc(WIDTH * sizeof(int64_t));
        if (!t->tail) {
            Py_DECREF(t);
            PyErr_NoMemory();
            return NULL;
        }
        memcpy(t->tail, self->tail, self->tail_len * sizeof(int64_t));
        t->tail_len = self->tail_len;
        t->tail_cap = WIDTH;
    } else {
        t->tail = (int64_t *)malloc(WIDTH * sizeof(int64_t));
        if (!t->tail) {
            Py_DECREF(t);
            PyErr_NoMemory();
            return NULL;
        }
        t->tail_len = 0;
        t->tail_cap = WIDTH;
    }

    return (PyObject *)t;
}

static int IntVector_init(IntVector *self, PyObject *args, PyObject *kwds) {
    if (kwds && PyDict_GET_SIZE(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "IntVector takes no keyword arguments");
        return -1;
    }

    Py_ssize_t n = PyTuple_Size(args);

    if (n == 0) {
        return 0;  // Empty vector already set up in __new__
    }

    // Check if single argument that's an iterable (but not a string)
    if (n == 1) {
        PyObject *arg = PyTuple_GET_ITEM(args, 0);
        if (!PyUnicode_Check(arg) && !PyBytes_Check(arg)) {
            PyObject *iter = PyObject_GetIter(arg);
            if (iter != NULL) {
                PyObject *item;
                while ((item = PyIter_Next(iter)) != NULL) {
                    PyObject *new_vec = IntVector_conj(self, item);
                    Py_DECREF(item);
                    if (!new_vec) {
                        Py_DECREF(iter);
                        return -1;
                    }

                    // Update self from new_vec
                    IntVector *nv = (IntVector *)new_vec;
                    Py_DECREF(self->root);
                    if (self->tail) free(self->tail);
                    self->cnt = nv->cnt;
                    self->shift = nv->shift;
                    self->root = nv->root;
                    Py_INCREF(self->root);
                    if (nv->tail && nv->tail_len > 0) {
                        self->tail = (int64_t *)malloc(nv->tail_len * sizeof(int64_t));
                        memcpy(self->tail, nv->tail, nv->tail_len * sizeof(int64_t));
                        self->tail_len = nv->tail_len;
                        self->tail_cap = nv->tail_len;
                    } else {
                        self->tail = NULL;
                        self->tail_len = 0;
                        self->tail_cap = 0;
                    }
                    Py_DECREF(new_vec);
                }
                Py_DECREF(iter);

                if (PyErr_Occurred()) return -1;
                return 0;
            }
            PyErr_Clear();
        }
    }

    // Multiple arguments or single non-iterable: treat as varargs
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        PyObject *new_vec = IntVector_conj(self, item);
        if (!new_vec) {
            return -1;
        }

        // Update self from new_vec
        IntVector *nv = (IntVector *)new_vec;
        Py_DECREF(self->root);
        if (self->tail) free(self->tail);
        self->cnt = nv->cnt;
        self->shift = nv->shift;
        self->root = nv->root;
        Py_INCREF(self->root);
        if (nv->tail && nv->tail_len > 0) {
            self->tail = (int64_t *)malloc(nv->tail_len * sizeof(int64_t));
            memcpy(self->tail, nv->tail, nv->tail_len * sizeof(int64_t));
            self->tail_len = nv->tail_len;
            self->tail_cap = nv->tail_len;
        } else {
            self->tail = NULL;
            self->tail_len = 0;
            self->tail_cap = 0;
        }
        Py_DECREF(new_vec);
    }

    return 0;
}

static PyObject *IntVector_reduce(IntVector *self, PyObject *Py_UNUSED(ignored)) {
    // Convert IntVector to a tuple using the sequence protocol
    PyObject *args = PySequence_Tuple((PyObject *)self);
    if (args == NULL) {
        return NULL;
    }

    // Return (type, args_tuple) - pickle will call type(*args_tuple)
    PyObject *result = PyTuple_Pack(2, (PyObject *)Py_TYPE(self), args);
    Py_DECREF(args);
    return result;
}

static PyMethodDef IntVector_methods[] = {
    {"nth", (PyCFunction)IntVector_nth, METH_VARARGS, "Get element at index"},
    {"conj", (PyCFunction)IntVector_conj, METH_O, "Add element to end"},
    {"transient", (PyCFunction)IntVector_transient, METH_NOARGS, "Return transient version for batch operations"},
    {"__reduce__", (PyCFunction)IntVector_reduce, METH_NOARGS, "Pickle support"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations"},
    {NULL}
};

static PyMappingMethods IntVector_as_mapping = {
    .mp_length = (lenfunc)IntVector_length,
    .mp_subscript = (binaryfunc)IntVector_getitem,
};

// Sequence protocol sq_item - takes Py_ssize_t index directly
static PyObject *IntVector_sq_item(IntVector *self, Py_ssize_t i) {
    if (i < 0) {
        i = self->cnt + i;
    }
    if (i < 0 || i >= self->cnt) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    int64_t val = IntVector_nth_raw(self, i);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLongLong(val);
}

static PySequenceMethods IntVector_as_sequence = {
    .sq_length = (lenfunc)IntVector_length,
    .sq_item = (ssizeargfunc)IntVector_sq_item,
};

static PyTypeObject IntVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.IntVector",
    .tp_doc = "Persistent Vector of longs (int64) with buffer protocol support",
    .tp_basicsize = sizeof(IntVector),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)IntVector_dealloc,
    .tp_repr = (reprfunc)IntVector_repr,
    .tp_as_buffer = &IntVector_as_buffer,
    .tp_as_sequence = &IntVector_as_sequence,
    .tp_as_mapping = &IntVector_as_mapping,
    .tp_hash = (hashfunc)IntVector_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = (getiterfunc)IntVector_iter,
    .tp_methods = IntVector_methods,
    .tp_init = (initproc)IntVector_init,
    .tp_new = IntVector_new,
};

// === TransientVector ===
typedef struct TransientVector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    VectorNode *root;
    PyObject *tail;  // list
    PyObject *id;
} TransientVector;

static void TransientVector_dealloc(TransientVector *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->tail);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Vector_transient(Vector *self, PyObject *Py_UNUSED(ignored)) {
    TransientVector *t = PyObject_New(TransientVector, &TransientVectorType);
    if (!t) return NULL;

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        return NULL;
    }

    t->cnt = self->cnt;
    t->shift = self->shift;
    t->root = VectorNode_clone(self->root, t->id);
    if (!t->root) {
        Py_DECREF(t);
        return NULL;
    }

    t->tail = PyList_New(PyTuple_Size(self->tail));
    if (!t->tail) {
        Py_DECREF(t);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < PyTuple_Size(self->tail); i++) {
        PyObject *item = PyTuple_GET_ITEM(self->tail, i);
        Py_INCREF(item);
        PyList_SET_ITEM(t->tail, i, item);
    }

    return (PyObject *)t;
}

static Py_ssize_t TransientVector_tail_off(TransientVector *self) {
    if (self->cnt < WIDTH) {
        return 0;
    }
    return ((self->cnt - 1) >> BITS) << BITS;
}

static void TransientVector_ensure_editable(TransientVector *self) {
    if (self->id == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Transient used after persistent() call");
    }
}

static VectorNode *TransientVector_new_path(TransientVector *self, int level, VectorNode *node) {
    if (level == 0) {
        Py_INCREF(node);
        return node;
    }
    VectorNode *ret = VectorNode_create(self->id);
    if (!ret) return NULL;

    VectorNode *child = TransientVector_new_path(self, level - BITS, node);
    if (!child) {
        Py_DECREF(ret);
        return NULL;
    }
    ret->array[0] = (PyObject *)child;
    return ret;
}

static VectorNode *TransientVector_push_tail(TransientVector *self, int level, VectorNode *parent, VectorNode *tail_node) {
    int subidx = ((self->cnt - 1) >> level) & MASK;
    VectorNode *ret;

    if (VectorNode_is_editable(parent, self->id)) {
        ret = parent;
        Py_INCREF(ret);
    } else {
        ret = VectorNode_clone(parent, self->id);
        if (!ret) return NULL;
    }

    VectorNode *node_to_insert;
    if (level == BITS) {
        node_to_insert = tail_node;
        Py_INCREF(tail_node);
    } else {
        PyObject *child = parent->array[subidx];
        if (child != NULL) {
            node_to_insert = TransientVector_push_tail(self, level - BITS, (VectorNode *)child, tail_node);
        } else {
            node_to_insert = TransientVector_new_path(self, level - BITS, tail_node);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    Py_XDECREF(ret->array[subidx]);
    ret->array[subidx] = (PyObject *)node_to_insert;
    return ret;
}

static PyObject *TransientVector_conj_mut(TransientVector *self, PyObject *val) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    // Room in tail?
    Py_ssize_t tail_len = PyList_Size(self->tail);
    if (self->cnt - TransientVector_tail_off(self) < WIDTH) {
        if (PyList_Append(self->tail, val) < 0) return NULL;
        self->cnt++;
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Tail is full, push into trie
    VectorNode *tail_node = VectorNode_create(self->id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < tail_len && i < WIDTH; i++) {
        tail_node->array[i] = PyList_GET_ITEM(self->tail, i);
        Py_INCREF(tail_node->array[i]);
    }

    // Reset tail
    Py_DECREF(self->tail);
    self->tail = PyList_New(1);
    if (!self->tail) {
        Py_DECREF(tail_node);
        return NULL;
    }
    Py_INCREF(val);
    PyList_SET_ITEM(self->tail, 0, val);

    // Overflow root?
    if (((size_t)self->cnt >> BITS) > (1ULL << self->shift)) {
        VectorNode *new_root = VectorNode_create(self->id);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->array[0] = (PyObject *)self->root;
        Py_INCREF(self->root);

        VectorNode *path = TransientVector_new_path(self, self->shift, tail_node);
        if (!path) {
            Py_DECREF(new_root);
            Py_DECREF(tail_node);
            return NULL;
        }
        new_root->array[1] = (PyObject *)path;

        Py_DECREF(self->root);
        self->root = new_root;
        self->shift += BITS;
    } else {
        VectorNode *new_root = TransientVector_push_tail(self, self->shift, self->root, tail_node);
        if (!new_root) {
            Py_DECREF(tail_node);
            return NULL;
        }
        Py_DECREF(self->root);
        self->root = new_root;
    }

    Py_DECREF(tail_node);
    self->cnt++;

    Py_INCREF(self);
    return (PyObject *)self;
}

static VectorNode *TransientVector_do_assoc(TransientVector *self, int level, VectorNode *node, Py_ssize_t i, PyObject *val) {
    VectorNode *ret;

    if (VectorNode_is_editable(node, self->id)) {
        ret = node;
        Py_INCREF(ret);
    } else {
        ret = VectorNode_clone(node, self->id);
        if (!ret) return NULL;
    }

    if (level == 0) {
        Py_XDECREF(ret->array[i & MASK]);
        ret->array[i & MASK] = val;
        Py_INCREF(val);
    } else {
        int subidx = (i >> level) & MASK;
        VectorNode *child = TransientVector_do_assoc(self, level - BITS, (VectorNode *)node->array[subidx], i, val);
        if (!child) {
            Py_DECREF(ret);
            return NULL;
        }
        Py_XDECREF(ret->array[subidx]);
        ret->array[subidx] = (PyObject *)child;
    }
    return ret;
}

static PyObject *TransientVector_assoc_mut(TransientVector *self, PyObject *args) {
    Py_ssize_t i;
    PyObject *val;

    if (!PyArg_ParseTuple(args, "nO", &i, &val)) {
        return NULL;
    }

    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (i < 0) {
        i = self->cnt + i;
    }

    if (i < 0 || i > self->cnt) {
        PyErr_Format(PyExc_IndexError, "Index %zd out of range", i);
        return NULL;
    }

    if (i == self->cnt) {
        return TransientVector_conj_mut(self, val);
    }

    Py_ssize_t tail_off = TransientVector_tail_off(self);
    if (i >= tail_off) {
        // Update in tail
        Py_ssize_t tail_idx = i - tail_off;
        PyObject *old = PyList_GET_ITEM(self->tail, tail_idx);
        Py_INCREF(val);
        PyList_SET_ITEM(self->tail, tail_idx, val);
        Py_DECREF(old);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Update in trie
    VectorNode *new_root = TransientVector_do_assoc(self, self->shift, self->root, i, val);
    if (!new_root) return NULL;

    if (new_root != self->root) {
        Py_DECREF(self->root);
        self->root = new_root;
    } else {
        Py_DECREF(new_root);
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

static VectorNode *TransientVector_pop_tail(TransientVector *self, int level, VectorNode *node) {
    int subidx = ((self->cnt - 2) >> level) & MASK;
    VectorNode *ret;

    if (VectorNode_is_editable(node, self->id)) {
        ret = node;
        Py_INCREF(ret);
    } else {
        ret = VectorNode_clone(node, self->id);
        if (!ret) return NULL;
    }

    if (level > BITS) {
        VectorNode *new_child = TransientVector_pop_tail(self, level - BITS, (VectorNode *)node->array[subidx]);
        if (new_child == NULL && !PyErr_Occurred()) {
            // Child became empty
            if (subidx == 0) {
                Py_DECREF(ret);
                return NULL;
            }
            Py_XDECREF(ret->array[subidx]);
            ret->array[subidx] = NULL;
        } else if (new_child == NULL) {
            Py_DECREF(ret);
            return NULL;
        } else {
            Py_XDECREF(ret->array[subidx]);
            ret->array[subidx] = (PyObject *)new_child;
        }
        return ret;
    } else if (subidx == 0) {
        Py_DECREF(ret);
        return NULL;
    } else {
        Py_XDECREF(ret->array[subidx]);
        ret->array[subidx] = NULL;
        return ret;
    }
}

static PyObject *TransientVector_pop_mut(TransientVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->cnt == 0) {
        PyErr_SetString(PyExc_IndexError, "Can't pop from empty vector");
        return NULL;
    }

    if (self->cnt == 1) {
        self->cnt = 0;
        Py_DECREF(self->tail);
        self->tail = PyList_New(0);
        if (!self->tail) return NULL;
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_ssize_t tail_len = PyList_Size(self->tail);
    if (tail_len > 1) {
        // Just remove from tail
        if (PyList_SetSlice(self->tail, tail_len - 1, tail_len, NULL) < 0) {
            return NULL;
        }
        self->cnt--;
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Tail has only one element, need to get new tail from trie

    // Find the new tail in the trie
    VectorNode *node = self->root;
    for (int level = self->shift; level > 0; level -= BITS) {
        node = (VectorNode *)node->array[((self->cnt - 2) >> level) & MASK];
    }

    // Create new tail from the leaf node
    Py_DECREF(self->tail);
    self->tail = PyList_New(WIDTH);
    if (!self->tail) return NULL;

    for (int i = 0; i < WIDTH; i++) {
        PyObject *item = node->array[i];
        if (item) {
            Py_INCREF(item);
            PyList_SET_ITEM(self->tail, i, item);
        } else {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(self->tail, i, Py_None);
        }
    }

    // Remove the last leaf from trie
    VectorNode *new_root = TransientVector_pop_tail(self, self->shift, self->root);
    int new_shift = self->shift;

    if (new_root == NULL && !PyErr_Occurred()) {
        new_root = VectorNode_create(self->id);
        if (!new_root) return NULL;
    } else if (new_root == NULL) {
        return NULL;
    }

    // Check if we can reduce depth
    if (self->shift > BITS && new_root->array[1] == NULL) {
        VectorNode *nr = (VectorNode *)new_root->array[0];
        Py_INCREF(nr);
        Py_DECREF(new_root);
        new_root = nr;
        new_shift -= BITS;
    }

    Py_DECREF(self->root);
    self->root = new_root;
    self->shift = new_shift;
    self->cnt--;

    // Trim tail to actual size
    Py_ssize_t actual_tail_size = (self->cnt - 1) & MASK;
    if (actual_tail_size < WIDTH - 1) {
        if (PyList_SetSlice(self->tail, actual_tail_size + 1, WIDTH, NULL) < 0) {
            return NULL;
        }
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientVector_persistent(TransientVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_CLEAR(self->id);

    PyObject *tail_tuple = PyList_AsTuple(self->tail);
    if (!tail_tuple) return NULL;

    Vector *result = Vector_create(self->cnt, self->shift, self->root, tail_tuple, NULL);
    Py_DECREF(tail_tuple);
    return (PyObject *)result;
}

// === TransientVector MutableSequence Protocol ===

static Py_ssize_t TransientVector_length(TransientVector *self) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return -1;
    return self->cnt;
}

static PyObject *TransientVector_sq_item(TransientVector *self, Py_ssize_t i) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (i < 0) {
        i = self->cnt + i;
    }
    if (i < 0 || i >= self->cnt) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Py_ssize_t tail_off = TransientVector_tail_off(self);
    if (i >= tail_off) {
        // In tail (which is a PyList)
        PyObject *result = PyList_GET_ITEM(self->tail, i - tail_off);
        Py_INCREF(result);
        return result;
    }

    // In trie - navigate to the leaf node
    VectorNode *node = self->root;
    for (int level = self->shift; level > 0; level -= BITS) {
        node = (VectorNode *)node->array[(i >> level) & MASK];
    }
    PyObject *result = node->array[i & MASK];
    Py_INCREF(result);
    return result;
}

static PyObject *TransientVector_getitem(TransientVector *self, PyObject *key) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (PyIndex_Check(key)) {
        Py_ssize_t i = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred()) return NULL;
        return TransientVector_sq_item(self, i);
    }

    PyErr_SetString(PyExc_TypeError, "indices must be integers");
    return NULL;
}

static int TransientVector_sq_ass_item(TransientVector *self, Py_ssize_t i, PyObject *val) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return -1;

    if (i < 0) {
        i = self->cnt + i;
    }

    if (val != NULL) {
        // Set: t[i] = v
        if (i < 0 || i >= self->cnt) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }

        Py_ssize_t tail_off = TransientVector_tail_off(self);
        if (i >= tail_off) {
            // Update in tail
            PyObject *old = PyList_GET_ITEM(self->tail, i - tail_off);
            Py_INCREF(val);
            PyList_SET_ITEM(self->tail, i - tail_off, val);
            Py_DECREF(old);
            return 0;
        }

        // Update in trie
        VectorNode *new_root = TransientVector_do_assoc(self, self->shift, self->root, i, val);
        if (!new_root) return -1;

        if (new_root != self->root) {
            Py_DECREF(self->root);
            self->root = new_root;
        } else {
            Py_DECREF(new_root);
        }
        return 0;
    } else {
        // Delete: del t[i]
        // TransientVector only supports popping from the end
        if (i < 0 || i >= self->cnt) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }

        if (i == self->cnt - 1) {
            PyObject *result = TransientVector_pop_mut(self, NULL);
            if (!result) return -1;
            Py_DECREF(result);
            return 0;
        } else {
            PyErr_SetString(PyExc_NotImplementedError,
                "TransientVector only supports deleting from the end (use pop_mut)");
            return -1;
        }
    }
}

static int TransientVector_contains(TransientVector *self, PyObject *val) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return -1;

    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        PyObject *item = TransientVector_sq_item(self, i);
        if (!item) return -1;

        int cmp = PyObject_RichCompareBool(item, val, Py_EQ);
        Py_DECREF(item);

        if (cmp < 0) return -1;  // Error
        if (cmp) return 1;       // Found
    }
    return 0;  // Not found
}

// TransientVector iterator
typedef struct {
    PyObject_HEAD
    TransientVector *tvec;
    Py_ssize_t index;
} TransientVectorIterator;

static PyTypeObject TransientVectorIteratorType;

static void TransientVectorIterator_dealloc(TransientVectorIterator *self) {
    Py_XDECREF(self->tvec);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *TransientVectorIterator_next(TransientVectorIterator *self) {
    if (self->tvec->id == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Transient used after persistent() call");
        return NULL;
    }

    if (self->index >= self->tvec->cnt) {
        return NULL;  // StopIteration
    }

    PyObject *result = TransientVector_sq_item(self->tvec, self->index);
    self->index++;
    return result;
}

static PyTypeObject TransientVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientVectorIterator",
    .tp_basicsize = sizeof(TransientVectorIterator),
    .tp_dealloc = (destructor)TransientVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)TransientVectorIterator_next,
};

static PyObject *TransientVector_iter(TransientVector *self) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    TransientVectorIterator *it = PyObject_New(TransientVectorIterator, &TransientVectorIteratorType);
    if (!it) return NULL;

    it->tvec = self;
    Py_INCREF(self);
    it->index = 0;
    return (PyObject *)it;
}

static PyObject *TransientVector_append(TransientVector *self, PyObject *val) {
    return TransientVector_conj_mut(self, val);
}

static PyObject *TransientVector_extend(TransientVector *self, PyObject *iterable) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    PyObject *iter = PyObject_GetIter(iterable);
    if (!iter) return NULL;

    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        PyObject *result = TransientVector_conj_mut(self, item);
        Py_DECREF(item);
        if (!result) {
            Py_DECREF(iter);
            return NULL;
        }
        Py_DECREF(result);
    }

    Py_DECREF(iter);
    if (PyErr_Occurred()) return NULL;

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientVector_sort(TransientVector *self, PyObject *args, PyObject *kwargs) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_ssize_t len = self->cnt;
    if (len <= 1) {
        Py_RETURN_NONE;  // Already sorted
    }

    // Extract all elements into a Python list
    PyObject *list = PyList_New(len);
    if (!list) return NULL;

    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = TransientVector_sq_item(self, i);
        if (!item) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, item);  // Steals reference
    }

    // Sort the list using Python's timsort
    PyObject *sort_method = PyObject_GetAttrString(list, "sort");
    if (!sort_method) {
        Py_DECREF(list);
        return NULL;
    }

    PyObject *empty_args = PyTuple_New(0);
    if (!empty_args) {
        Py_DECREF(sort_method);
        Py_DECREF(list);
        return NULL;
    }

    PyObject *sort_result = PyObject_Call(sort_method, empty_args, kwargs);
    Py_DECREF(empty_args);
    Py_DECREF(sort_method);
    if (!sort_result) {
        Py_DECREF(list);
        return NULL;
    }
    Py_DECREF(sort_result);

    // Rebuild the transient vector from scratch with sorted elements
    // Create a fresh edit id
    PyObject *new_id = PyObject_New(PyObject, &PdsSentinelType);
    if (!new_id) {
        Py_DECREF(list);
        return NULL;
    }

    // Create new empty root
    VectorNode *new_root = VectorNode_create(new_id);
    if (!new_root) {
        Py_DECREF(new_id);
        Py_DECREF(list);
        return NULL;
    }

    // Create new empty tail
    PyObject *new_tail = PyList_New(0);
    if (!new_tail) {
        Py_DECREF(new_root);
        Py_DECREF(new_id);
        Py_DECREF(list);
        return NULL;
    }

    // Replace old structures
    Py_DECREF(self->root);
    Py_DECREF(self->tail);
    Py_DECREF(self->id);
    self->root = new_root;
    self->tail = new_tail;
    self->id = new_id;
    self->cnt = 0;
    self->shift = BITS;

    // Add all sorted elements back
    for (Py_ssize_t i = 0; i < len; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        PyObject *result = TransientVector_conj_mut(self, item);
        if (!result) {
            Py_DECREF(list);
            return NULL;
        }
        Py_DECREF(result);
    }

    Py_DECREF(list);
    Py_RETURN_NONE;
}

static PyMethodDef TransientVector_methods[] = {
    {"conj_mut", (PyCFunction)TransientVector_conj_mut, METH_O, "Mutably add element"},
    {"assoc_mut", (PyCFunction)TransientVector_assoc_mut, METH_VARARGS, "Mutably set element at index"},
    {"pop_mut", (PyCFunction)TransientVector_pop_mut, METH_NOARGS, "Mutably remove last element"},
    {"persistent", (PyCFunction)TransientVector_persistent, METH_NOARGS, "Return persistent vector"},
    {"append", (PyCFunction)TransientVector_append, METH_O, "Append element (alias for conj_mut)"},
    {"extend", (PyCFunction)TransientVector_extend, METH_O, "Extend with elements from iterable"},
    {"sort", (PyCFunction)TransientVector_sort, METH_VARARGS | METH_KEYWORDS, "Sort elements in place"},
    {NULL}
};

static PySequenceMethods TransientVector_as_sequence = {
    .sq_length = (lenfunc)TransientVector_length,
    .sq_item = (ssizeargfunc)TransientVector_sq_item,
    .sq_ass_item = (ssizeobjargproc)TransientVector_sq_ass_item,
    .sq_contains = (objobjproc)TransientVector_contains,
};

static PyMappingMethods TransientVector_as_mapping = {
    .mp_length = (lenfunc)TransientVector_length,
    .mp_subscript = (binaryfunc)TransientVector_getitem,
};

static PyTypeObject TransientVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientVector",
    .tp_doc = "Transient vector for batch operations",
    .tp_basicsize = sizeof(TransientVector),
    .tp_dealloc = (destructor)TransientVector_dealloc,
    .tp_as_sequence = &TransientVector_as_sequence,
    .tp_as_mapping = &TransientVector_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = (getiterfunc)TransientVector_iter,
    .tp_methods = TransientVector_methods,
};

// === Map Nodes ===

// Forward declarations for iterator functions
static PyObject *BitmapIndexedNode_iter_mode(BitmapIndexedNode *self, int mode);
static PyObject *ArrayNode_iter_mode(ArrayNode *self, int mode);
static PyObject *HashCollisionNode_iter_mode(HashCollisionNode *self, int mode);

// BitmapIndexedNode
typedef struct BitmapIndexedNode {
    PyObject_HEAD
    unsigned int bitmap;
    PyObject *array;  // list
    PyObject *transient_id;
} BitmapIndexedNode;

static PyTypeObject BitmapIndexedNodeType;
static BitmapIndexedNode *EMPTY_BIN = NULL;

static void BitmapIndexedNode_dealloc(BitmapIndexedNode *self) {
    Py_XDECREF(self->array);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static BitmapIndexedNode *BitmapIndexedNode_create(unsigned int bitmap, PyObject *array, PyObject *transient_id) {
    BitmapIndexedNode *node = PyObject_New(BitmapIndexedNode, &BitmapIndexedNodeType);
    if (!node) return NULL;

    node->bitmap = bitmap;
    node->array = array ? array : PyList_New(0);
    if (!node->array) {
        Py_DECREF(node);
        return NULL;
    }
    if (array) Py_INCREF(array);
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);

    return node;
}

static int BitmapIndexedNode_is_editable(BitmapIndexedNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

static BitmapIndexedNode *BitmapIndexedNode_ensure_editable(BitmapIndexedNode *self, PyObject *transient_id) {
    if (BitmapIndexedNode_is_editable(self, transient_id)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *new_array = PyList_GetSlice(self->array, 0, PyList_Size(self->array));
    if (!new_array) return NULL;
    BitmapIndexedNode *result = BitmapIndexedNode_create(self->bitmap, new_array, transient_id);
    Py_DECREF(new_array);
    return result;
}

// Forward declarations for node operations
static PyObject *BitmapIndexedNode_assoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id);
static PyObject *BitmapIndexedNode_find(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found);
static PyObject *BitmapIndexedNode_dissoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id);
static PyObject *BitmapIndexedNode_iter_kv(BitmapIndexedNode *self);

// ArrayNode
typedef struct ArrayNode {
    PyObject_HEAD
    int count;
    PyObject *array;  // list of WIDTH nodes
    PyObject *transient_id;
} ArrayNode;

static PyTypeObject ArrayNodeType;

static void ArrayNode_dealloc(ArrayNode *self) {
    Py_XDECREF(self->array);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static ArrayNode *ArrayNode_create(int count, PyObject *array, PyObject *transient_id) {
    ArrayNode *node = PyObject_New(ArrayNode, &ArrayNodeType);
    if (!node) return NULL;

    node->count = count;
    if (array) {
        node->array = array;
        Py_INCREF(array);
    } else {
        node->array = PyList_New(WIDTH);
        if (!node->array) {
            Py_DECREF(node);
            return NULL;
        }
        for (int i = 0; i < WIDTH; i++) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(node->array, i, Py_None);
        }
    }
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);

    return node;
}

static int ArrayNode_is_editable(ArrayNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

static ArrayNode *ArrayNode_ensure_editable(ArrayNode *self, PyObject *transient_id) {
    if (ArrayNode_is_editable(self, transient_id)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *new_array = PyList_GetSlice(self->array, 0, WIDTH);
    if (!new_array) return NULL;
    ArrayNode *result = ArrayNode_create(self->count, new_array, transient_id);
    Py_DECREF(new_array);
    return result;
}

static PyObject *ArrayNode_assoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id);
static PyObject *ArrayNode_find(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found);
static PyObject *ArrayNode_dissoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id);
static PyObject *ArrayNode_iter_kv(ArrayNode *self);

// HashCollisionNode
typedef struct HashCollisionNode {
    PyObject_HEAD
    Py_hash_t hash;
    int count;
    PyObject *array;  // list [k1, v1, k2, v2, ...]
    PyObject *transient_id;
} HashCollisionNode;

static PyTypeObject HashCollisionNodeType;

static void HashCollisionNode_dealloc(HashCollisionNode *self) {
    Py_XDECREF(self->array);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static HashCollisionNode *HashCollisionNode_create(Py_hash_t hash_val, int count, PyObject *array, PyObject *transient_id) {
    HashCollisionNode *node = PyObject_New(HashCollisionNode, &HashCollisionNodeType);
    if (!node) return NULL;

    node->hash = hash_val;
    node->count = count;
    node->array = array ? array : PyList_New(0);
    if (!node->array) {
        Py_DECREF(node);
        return NULL;
    }
    if (array) Py_INCREF(array);
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);

    return node;
}

static int HashCollisionNode_is_editable(HashCollisionNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

static HashCollisionNode *HashCollisionNode_ensure_editable(HashCollisionNode *self, PyObject *transient_id) {
    if (HashCollisionNode_is_editable(self, transient_id)) {
        Py_INCREF(self);
        return self;
    }
    PyObject *new_array = PyList_GetSlice(self->array, 0, PyList_Size(self->array));
    if (!new_array) return NULL;
    HashCollisionNode *result = HashCollisionNode_create(self->hash, self->count, new_array, transient_id);
    Py_DECREF(new_array);
    return result;
}

static int HashCollisionNode_find_index(HashCollisionNode *self, PyObject *key) {
    for (int i = 0; i < 2 * self->count; i += 2) {
        PyObject *k = PyList_GET_ITEM(self->array, i);
        int eq = PyObject_RichCompareBool(k, key, Py_EQ);
        if (eq < 0) return -2;  // Error
        if (eq) return i;
    }
    return -1;
}

static PyObject *HashCollisionNode_assoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id);
static PyObject *HashCollisionNode_find(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found);
static PyObject *HashCollisionNode_dissoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id);
static PyObject *HashCollisionNode_iter_kv(HashCollisionNode *self);

// Node type definitions
static PyTypeObject BitmapIndexedNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.BitmapIndexedNode",
    .tp_basicsize = sizeof(BitmapIndexedNode),
    .tp_dealloc = (destructor)BitmapIndexedNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

static PyTypeObject ArrayNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.ArrayNode",
    .tp_basicsize = sizeof(ArrayNode),
    .tp_dealloc = (destructor)ArrayNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

static PyTypeObject HashCollisionNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.HashCollisionNode",
    .tp_basicsize = sizeof(HashCollisionNode),
    .tp_dealloc = (destructor)HashCollisionNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// Helper to create a node for two key-value pairs
static PyObject *create_node(int shift, PyObject *key1, PyObject *val1, Py_hash_t hash2, PyObject *key2, PyObject *val2, PyObject *transient_id) {
    Py_hash_t hash1 = PyObject_Hash(key1);
    if (hash1 == -1 && PyErr_Occurred()) return NULL;

    if (hash1 == hash2) {
        PyObject *arr = PyList_New(4);
        if (!arr) return NULL;
        Py_INCREF(key1); PyList_SET_ITEM(arr, 0, key1);
        Py_INCREF(val1); PyList_SET_ITEM(arr, 1, val1);
        Py_INCREF(key2); PyList_SET_ITEM(arr, 2, key2);
        Py_INCREF(val2); PyList_SET_ITEM(arr, 3, val2);
        HashCollisionNode *node = HashCollisionNode_create(hash1, 2, arr, transient_id);
        Py_DECREF(arr);
        return (PyObject *)node;
    }

    PyObject *added_leaf = PyList_New(0);
    if (!added_leaf) return NULL;

    PyObject *n1 = BitmapIndexedNode_assoc(EMPTY_BIN, shift, hash1, key1, val1, added_leaf, transient_id);
    if (!n1) {
        Py_DECREF(added_leaf);
        return NULL;
    }

    PyObject *n2;
    if (PyObject_TypeCheck(n1, &BitmapIndexedNodeType)) {
        n2 = BitmapIndexedNode_assoc((BitmapIndexedNode *)n1, shift, hash2, key2, val2, added_leaf, transient_id);
    } else if (PyObject_TypeCheck(n1, &ArrayNodeType)) {
        n2 = ArrayNode_assoc((ArrayNode *)n1, shift, hash2, key2, val2, added_leaf, transient_id);
    } else {
        n2 = HashCollisionNode_assoc((HashCollisionNode *)n1, shift, hash2, key2, val2, added_leaf, transient_id);
    }

    Py_DECREF(added_leaf);
    Py_DECREF(n1);
    return n2;
}

// BitmapIndexedNode implementation
static PyObject *BitmapIndexedNode_assoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id) {
    unsigned int bit = bitpos(hash_val, shift);
    int idx = bitmap_index(self->bitmap, bit);

    if (self->bitmap & bit) {
        // Slot exists
        PyObject *key_or_null = PyList_GET_ITEM(self->array, 2 * idx);
        PyObject *val_or_node = PyList_GET_ITEM(self->array, 2 * idx + 1);

        if (key_or_null == Py_None) {
            // Child node
            PyObject *n;
            if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
                n = BitmapIndexedNode_assoc((BitmapIndexedNode *)val_or_node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
                n = ArrayNode_assoc((ArrayNode *)val_or_node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            } else {
                n = HashCollisionNode_assoc((HashCollisionNode *)val_or_node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            }
            if (!n) return NULL;

            if (n == val_or_node) {
                Py_DECREF(n);
                Py_INCREF(self);
                return (PyObject *)self;
            }

            BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
            if (!node) {
                Py_DECREF(n);
                return NULL;
            }
            PyList_SET_ITEM(node->array, 2 * idx + 1, n);
            return (PyObject *)node;
        }

        int eq = PyObject_RichCompareBool(key, key_or_null, Py_EQ);
        if (eq < 0) return NULL;

        if (eq) {
            // Same key - update value
            if (val == val_or_node) {
                Py_INCREF(self);
                return (PyObject *)self;
            }
            BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
            if (!node) return NULL;
            Py_INCREF(val);
            Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx + 1));
            PyList_SET_ITEM(node->array, 2 * idx + 1, val);
            return (PyObject *)node;
        }

        // Hash collision at this level - need to go deeper
        if (PyList_Append(added_leaf, Py_True) < 0) return NULL;

        PyObject *new_node = create_node(shift + BITS, key_or_null, val_or_node, hash_val, key, val, transient_id);
        if (!new_node) return NULL;

        BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
        if (!node) {
            Py_DECREF(new_node);
            return NULL;
        }
        Py_INCREF(Py_None);
        Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx));
        PyList_SET_ITEM(node->array, 2 * idx, Py_None);
        Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx + 1));
        PyList_SET_ITEM(node->array, 2 * idx + 1, new_node);
        return (PyObject *)node;
    } else {
        // New slot
        int n = ctpop(self->bitmap);
        if (n >= WIDTH / 2) {
            // Upgrade to ArrayNode
            PyObject *nodes = PyList_New(WIDTH);
            if (!nodes) return NULL;
            for (int i = 0; i < WIDTH; i++) {
                Py_INCREF(Py_None);
                PyList_SET_ITEM(nodes, i, Py_None);
            }

            int jdx = mask_hash(hash_val, shift);
            PyObject *new_bin = BitmapIndexedNode_assoc(EMPTY_BIN, shift + BITS, hash_val, key, val, added_leaf, transient_id);
            if (!new_bin) {
                Py_DECREF(nodes);
                return NULL;
            }
            Py_DECREF(PyList_GET_ITEM(nodes, jdx));
            PyList_SET_ITEM(nodes, jdx, new_bin);

            int j = 0;
            for (int i = 0; i < WIDTH; i++) {
                if ((self->bitmap >> i) & 1) {
                    PyObject *k = PyList_GET_ITEM(self->array, j);
                    PyObject *v = PyList_GET_ITEM(self->array, j + 1);
                    if (k == Py_None) {
                        Py_INCREF(v);
                        Py_DECREF(PyList_GET_ITEM(nodes, i));
                        PyList_SET_ITEM(nodes, i, v);
                    } else {
                        PyObject *al = PyList_New(0);
                        Py_hash_t kh = PyObject_Hash(k);
                        if (kh == -1 && PyErr_Occurred()) {
                            Py_DECREF(al);
                            Py_DECREF(nodes);
                            return NULL;
                        }
                        PyObject *child = BitmapIndexedNode_assoc(EMPTY_BIN, shift + BITS, kh, k, v, al, transient_id);
                        Py_DECREF(al);
                        if (!child) {
                            Py_DECREF(nodes);
                            return NULL;
                        }
                        Py_DECREF(PyList_GET_ITEM(nodes, i));
                        PyList_SET_ITEM(nodes, i, child);
                    }
                    j += 2;
                }
            }

            ArrayNode *result = ArrayNode_create(n + 1, nodes, transient_id);
            Py_DECREF(nodes);
            return (PyObject *)result;
        } else {
            // Insert into bitmap node
            if (PyList_Append(added_leaf, Py_True) < 0) return NULL;

            Py_ssize_t arr_len = PyList_Size(self->array);
            PyObject *new_array = PyList_New(arr_len + 2);
            if (!new_array) return NULL;

            for (Py_ssize_t i = 0; i < 2 * idx; i++) {
                PyObject *item = PyList_GET_ITEM(self->array, i);
                Py_INCREF(item);
                PyList_SET_ITEM(new_array, i, item);
            }
            Py_INCREF(key);
            PyList_SET_ITEM(new_array, 2 * idx, key);
            Py_INCREF(val);
            PyList_SET_ITEM(new_array, 2 * idx + 1, val);
            for (Py_ssize_t i = 2 * idx; i < arr_len; i++) {
                PyObject *item = PyList_GET_ITEM(self->array, i);
                Py_INCREF(item);
                PyList_SET_ITEM(new_array, i + 2, item);
            }

            BitmapIndexedNode *node = BitmapIndexedNode_create(self->bitmap | bit, new_array, transient_id);
            Py_DECREF(new_array);
            return (PyObject *)node;
        }
    }
}

static PyObject *BitmapIndexedNode_find(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found) {
    unsigned int bit = bitpos(hash_val, shift);
    if ((self->bitmap & bit) == 0) {
        Py_INCREF(not_found);
        return not_found;
    }

    int idx = bitmap_index(self->bitmap, bit);
    PyObject *key_or_null = PyList_GET_ITEM(self->array, 2 * idx);
    PyObject *val_or_node = PyList_GET_ITEM(self->array, 2 * idx + 1);

    if (key_or_null == Py_None) {
        if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
            return BitmapIndexedNode_find((BitmapIndexedNode *)val_or_node, shift + BITS, hash_val, key, not_found);
        } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
            return ArrayNode_find((ArrayNode *)val_or_node, shift + BITS, hash_val, key, not_found);
        } else {
            return HashCollisionNode_find((HashCollisionNode *)val_or_node, shift + BITS, hash_val, key, not_found);
        }
    }

    int eq = PyObject_RichCompareBool(key, key_or_null, Py_EQ);
    if (eq < 0) return NULL;

    if (eq) {
        Py_INCREF(val_or_node);
        return val_or_node;
    }

    Py_INCREF(not_found);
    return not_found;
}

static PyObject *BitmapIndexedNode_dissoc(BitmapIndexedNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id) {
    unsigned int bit = bitpos(hash_val, shift);
    if ((self->bitmap & bit) == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    int idx = bitmap_index(self->bitmap, bit);
    PyObject *key_or_null = PyList_GET_ITEM(self->array, 2 * idx);
    PyObject *val_or_node = PyList_GET_ITEM(self->array, 2 * idx + 1);

    if (key_or_null == Py_None) {
        // Recurse into child node
        PyObject *n;
        if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
            n = BitmapIndexedNode_dissoc((BitmapIndexedNode *)val_or_node, shift + BITS, hash_val, key, removed_leaf, transient_id);
        } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
            n = ArrayNode_dissoc((ArrayNode *)val_or_node, shift + BITS, hash_val, key, removed_leaf, transient_id);
        } else {
            n = HashCollisionNode_dissoc((HashCollisionNode *)val_or_node, shift + BITS, hash_val, key, removed_leaf, transient_id);
        }
        if (!n) return NULL;

        if (n == val_or_node) {
            Py_DECREF(n);
            Py_INCREF(self);
            return (PyObject *)self;
        }
        if (n != Py_None) {
            BitmapIndexedNode *node = BitmapIndexedNode_ensure_editable(self, transient_id);
            if (!node) {
                Py_DECREF(n);
                return NULL;
            }
            Py_DECREF(PyList_GET_ITEM(node->array, 2 * idx + 1));
            PyList_SET_ITEM(node->array, 2 * idx + 1, n);
            return (PyObject *)node;
        }
        if (self->bitmap == bit) {
            Py_DECREF(n);
            Py_INCREF(Py_None);
            return Py_None;
        }
        Py_DECREF(n);

        // Remove entry
        Py_ssize_t arr_len = PyList_Size(self->array);
        PyObject *new_array = PyList_New(arr_len - 2);
        if (!new_array) return NULL;

        for (Py_ssize_t i = 0; i < 2 * idx; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i, item);
        }
        for (Py_ssize_t i = 2 * idx + 2; i < arr_len; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i - 2, item);
        }

        BitmapIndexedNode *node = BitmapIndexedNode_create(self->bitmap ^ bit, new_array, transient_id);
        Py_DECREF(new_array);
        return (PyObject *)node;
    }

    int eq = PyObject_RichCompareBool(key, key_or_null, Py_EQ);
    if (eq < 0) return NULL;

    if (eq) {
        // Mark that we found and removed a leaf
        if (removed_leaf && PyList_Append(removed_leaf, Py_True) < 0) return NULL;

        if (self->bitmap == bit) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        Py_ssize_t arr_len = PyList_Size(self->array);
        PyObject *new_array = PyList_New(arr_len - 2);
        if (!new_array) return NULL;

        for (Py_ssize_t i = 0; i < 2 * idx; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i, item);
        }
        for (Py_ssize_t i = 2 * idx + 2; i < arr_len; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i - 2, item);
        }

        BitmapIndexedNode *node = BitmapIndexedNode_create(self->bitmap ^ bit, new_array, transient_id);
        Py_DECREF(new_array);
        return (PyObject *)node;
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

// Iterator mode constants
#define ITER_MODE_ITEMS 0
#define ITER_MODE_KEYS 1
#define ITER_MODE_VALUES 2

// BitmapIndexedNode iterator
typedef struct {
    PyObject_HEAD
    BitmapIndexedNode *node;
    Py_ssize_t index;
    PyObject *child_iter;
    int mode;  // ITER_MODE_ITEMS, ITER_MODE_KEYS, or ITER_MODE_VALUES
} BitmapIndexedNodeIterator;

static PyTypeObject BitmapIndexedNodeIteratorType;

static void BitmapIndexedNodeIterator_dealloc(BitmapIndexedNodeIterator *self) {
    Py_XDECREF(self->node);
    Py_XDECREF(self->child_iter);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *BitmapIndexedNodeIterator_next(BitmapIndexedNodeIterator *self) {
    // If we have a child iterator, try to get next from it
    while (self->child_iter) {
        PyObject *result = PyIter_Next(self->child_iter);
        if (result) return result;
        if (PyErr_Occurred()) return NULL;
        Py_CLEAR(self->child_iter);
    }

    Py_ssize_t arr_len = PyList_Size(self->node->array);
    while (self->index < arr_len) {
        PyObject *key_or_null = PyList_GET_ITEM(self->node->array, self->index);
        PyObject *val_or_node = PyList_GET_ITEM(self->node->array, self->index + 1);
        self->index += 2;

        if (key_or_null != Py_None) {
            // Direct key-value pair - return based on mode
            PyObject *result;
            switch (self->mode) {
                case ITER_MODE_KEYS:
                    result = key_or_null;
                    Py_INCREF(result);
                    break;
                case ITER_MODE_VALUES:
                    result = val_or_node;
                    Py_INCREF(result);
                    break;
                default:  // ITER_MODE_ITEMS
                    result = PyTuple_Pack(2, key_or_null, val_or_node);
                    break;
            }
            return result;
        } else if (val_or_node != Py_None) {
            // Child node - get its iterator with same mode
            if (PyObject_TypeCheck(val_or_node, &BitmapIndexedNodeType)) {
                self->child_iter = BitmapIndexedNode_iter_mode((BitmapIndexedNode *)val_or_node, self->mode);
            } else if (PyObject_TypeCheck(val_or_node, &ArrayNodeType)) {
                self->child_iter = ArrayNode_iter_mode((ArrayNode *)val_or_node, self->mode);
            } else {
                self->child_iter = HashCollisionNode_iter_mode((HashCollisionNode *)val_or_node, self->mode);
            }
            if (!self->child_iter) return NULL;

            PyObject *result = PyIter_Next(self->child_iter);
            if (result) return result;
            if (PyErr_Occurred()) return NULL;
            Py_CLEAR(self->child_iter);
        }
    }

    return NULL;  // StopIteration
}

static PyTypeObject BitmapIndexedNodeIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.BitmapIndexedNodeIterator",
    .tp_basicsize = sizeof(BitmapIndexedNodeIterator),
    .tp_dealloc = (destructor)BitmapIndexedNodeIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)BitmapIndexedNodeIterator_next,
};

static PyObject *BitmapIndexedNode_iter_mode(BitmapIndexedNode *self, int mode) {
    BitmapIndexedNodeIterator *it = PyObject_New(BitmapIndexedNodeIterator, &BitmapIndexedNodeIteratorType);
    if (!it) return NULL;

    it->node = self;
    Py_INCREF(self);
    it->index = 0;
    it->child_iter = NULL;
    it->mode = mode;
    return (PyObject *)it;
}

static PyObject *BitmapIndexedNode_iter_kv(BitmapIndexedNode *self) {
    return BitmapIndexedNode_iter_mode(self, ITER_MODE_ITEMS);
}

// ArrayNode implementation
static PyObject *ArrayNode_assoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id) {
    int idx = mask_hash(hash_val, shift);
    PyObject *node = PyList_GET_ITEM(self->array, idx);

    if (node == Py_None) {
        // Use a fresh added_leaf for recursive call to avoid double-counting
        // We'll count this insertion here, not in the recursive call
        PyObject *al_fresh = PyList_New(0);
        if (!al_fresh) return NULL;

        PyObject *new_node = BitmapIndexedNode_assoc(EMPTY_BIN, shift + BITS, hash_val, key, val, al_fresh, transient_id);
        Py_DECREF(al_fresh);
        if (!new_node) return NULL;

        // Count this as a new insertion
        if (PyList_Append(added_leaf, Py_True) < 0) {
            Py_DECREF(new_node);
            return NULL;
        }

        ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
        if (!editable) {
            Py_DECREF(new_node);
            return NULL;
        }
        Py_DECREF(PyList_GET_ITEM(editable->array, idx));
        PyList_SET_ITEM(editable->array, idx, new_node);
        editable->count++;
        return (PyObject *)editable;
    }

    PyObject *n;
    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        n = BitmapIndexedNode_assoc((BitmapIndexedNode *)node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
    } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        n = ArrayNode_assoc((ArrayNode *)node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
    } else {
        n = HashCollisionNode_assoc((HashCollisionNode *)node, shift + BITS, hash_val, key, val, added_leaf, transient_id);
    }
    if (!n) return NULL;

    if (n == node) {
        Py_DECREF(n);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
    if (!editable) {
        Py_DECREF(n);
        return NULL;
    }
    Py_DECREF(PyList_GET_ITEM(editable->array, idx));
    PyList_SET_ITEM(editable->array, idx, n);
    return (PyObject *)editable;
}

static PyObject *ArrayNode_find(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found) {
    int idx = mask_hash(hash_val, shift);
    PyObject *node = PyList_GET_ITEM(self->array, idx);

    if (node == Py_None) {
        Py_INCREF(not_found);
        return not_found;
    }

    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_find((BitmapIndexedNode *)node, shift + BITS, hash_val, key, not_found);
    } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        return ArrayNode_find((ArrayNode *)node, shift + BITS, hash_val, key, not_found);
    } else {
        return HashCollisionNode_find((HashCollisionNode *)node, shift + BITS, hash_val, key, not_found);
    }
}

static PyObject *ArrayNode_pack(ArrayNode *self, PyObject *transient_id, int idx) {
    PyObject *new_array = PyList_New(0);
    if (!new_array) return NULL;

    unsigned int bitmap = 0;
    for (int i = 0; i < WIDTH; i++) {
        PyObject *node = PyList_GET_ITEM(self->array, i);
        if (i != idx && node != Py_None) {
            Py_INCREF(Py_None);
            if (PyList_Append(new_array, Py_None) < 0) {
                Py_DECREF(new_array);
                return NULL;
            }
            Py_INCREF(node);
            if (PyList_Append(new_array, node) < 0) {
                Py_DECREF(new_array);
                return NULL;
            }
            bitmap |= 1U << i;
        }
    }

    BitmapIndexedNode *result = BitmapIndexedNode_create(bitmap, new_array, transient_id);
    Py_DECREF(new_array);
    return (PyObject *)result;
}

static PyObject *ArrayNode_dissoc(ArrayNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id) {
    int idx = mask_hash(hash_val, shift);
    PyObject *node = PyList_GET_ITEM(self->array, idx);

    if (node == Py_None) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    PyObject *n;
    if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
        n = BitmapIndexedNode_dissoc((BitmapIndexedNode *)node, shift + BITS, hash_val, key, removed_leaf, transient_id);
    } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
        n = ArrayNode_dissoc((ArrayNode *)node, shift + BITS, hash_val, key, removed_leaf, transient_id);
    } else {
        n = HashCollisionNode_dissoc((HashCollisionNode *)node, shift + BITS, hash_val, key, removed_leaf, transient_id);
    }
    if (!n) return NULL;

    if (n == node) {
        Py_DECREF(n);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    if (n == Py_None) {
        Py_DECREF(n);
        if (self->count <= WIDTH / 4) {
            return ArrayNode_pack(self, transient_id, idx);
        }
        ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
        if (!editable) return NULL;
        Py_INCREF(Py_None);
        Py_DECREF(PyList_GET_ITEM(editable->array, idx));
        PyList_SET_ITEM(editable->array, idx, Py_None);
        editable->count--;
        return (PyObject *)editable;
    }

    ArrayNode *editable = ArrayNode_ensure_editable(self, transient_id);
    if (!editable) {
        Py_DECREF(n);
        return NULL;
    }
    Py_DECREF(PyList_GET_ITEM(editable->array, idx));
    PyList_SET_ITEM(editable->array, idx, n);
    return (PyObject *)editable;
}

// ArrayNode iterator
typedef struct {
    PyObject_HEAD
    ArrayNode *node;
    int index;
    PyObject *child_iter;
    int mode;  // ITER_MODE_ITEMS, ITER_MODE_KEYS, or ITER_MODE_VALUES
} ArrayNodeIterator;

static PyTypeObject ArrayNodeIteratorType;

static void ArrayNodeIterator_dealloc(ArrayNodeIterator *self) {
    Py_XDECREF(self->node);
    Py_XDECREF(self->child_iter);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *ArrayNodeIterator_next(ArrayNodeIterator *self) {
    while (self->child_iter) {
        PyObject *result = PyIter_Next(self->child_iter);
        if (result) return result;
        if (PyErr_Occurred()) return NULL;
        Py_CLEAR(self->child_iter);
    }

    while (self->index < WIDTH) {
        PyObject *node = PyList_GET_ITEM(self->node->array, self->index);
        self->index++;

        if (node != Py_None) {
            if (PyObject_TypeCheck(node, &BitmapIndexedNodeType)) {
                self->child_iter = BitmapIndexedNode_iter_mode((BitmapIndexedNode *)node, self->mode);
            } else if (PyObject_TypeCheck(node, &ArrayNodeType)) {
                self->child_iter = ArrayNode_iter_mode((ArrayNode *)node, self->mode);
            } else {
                self->child_iter = HashCollisionNode_iter_mode((HashCollisionNode *)node, self->mode);
            }
            if (!self->child_iter) return NULL;

            PyObject *result = PyIter_Next(self->child_iter);
            if (result) return result;
            if (PyErr_Occurred()) return NULL;
            Py_CLEAR(self->child_iter);
        }
    }

    return NULL;
}

static PyTypeObject ArrayNodeIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.ArrayNodeIterator",
    .tp_basicsize = sizeof(ArrayNodeIterator),
    .tp_dealloc = (destructor)ArrayNodeIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)ArrayNodeIterator_next,
};

static PyObject *ArrayNode_iter_mode(ArrayNode *self, int mode) {
    ArrayNodeIterator *it = PyObject_New(ArrayNodeIterator, &ArrayNodeIteratorType);
    if (!it) return NULL;

    it->node = self;
    Py_INCREF(self);
    it->index = 0;
    it->child_iter = NULL;
    it->mode = mode;
    return (PyObject *)it;
}

static PyObject *ArrayNode_iter_kv(ArrayNode *self) {
    return ArrayNode_iter_mode(self, ITER_MODE_ITEMS);
}

// HashCollisionNode implementation
static PyObject *HashCollisionNode_assoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *val, PyObject *added_leaf, PyObject *transient_id) {
    if (hash_val == self->hash) {
        int idx = HashCollisionNode_find_index(self, key);
        if (idx == -2) return NULL;  // Error

        if (idx != -1) {
            PyObject *existing = PyList_GET_ITEM(self->array, idx + 1);
            if (existing == val) {
                Py_INCREF(self);
                return (PyObject *)self;
            }
            HashCollisionNode *node = HashCollisionNode_ensure_editable(self, transient_id);
            if (!node) return NULL;
            Py_INCREF(val);
            Py_DECREF(PyList_GET_ITEM(node->array, idx + 1));
            PyList_SET_ITEM(node->array, idx + 1, val);
            return (PyObject *)node;
        }

        if (PyList_Append(added_leaf, Py_True) < 0) return NULL;

        Py_ssize_t arr_len = PyList_Size(self->array);
        PyObject *new_array = PyList_New(arr_len + 2);
        if (!new_array) return NULL;

        for (Py_ssize_t i = 0; i < arr_len; i++) {
            PyObject *item = PyList_GET_ITEM(self->array, i);
            Py_INCREF(item);
            PyList_SET_ITEM(new_array, i, item);
        }
        Py_INCREF(key);
        PyList_SET_ITEM(new_array, arr_len, key);
        Py_INCREF(val);
        PyList_SET_ITEM(new_array, arr_len + 1, val);

        HashCollisionNode *node = HashCollisionNode_create(self->hash, self->count + 1, new_array, transient_id);
        Py_DECREF(new_array);
        return (PyObject *)node;
    }

    // Different hash - nest in a bitmap node
    PyObject *arr = PyList_New(2);
    if (!arr) return NULL;
    Py_INCREF(Py_None);
    PyList_SET_ITEM(arr, 0, Py_None);
    Py_INCREF(self);
    PyList_SET_ITEM(arr, 1, (PyObject *)self);

    BitmapIndexedNode *bin = BitmapIndexedNode_create(bitpos(self->hash, shift), arr, transient_id);
    Py_DECREF(arr);
    if (!bin) return NULL;

    PyObject *result = BitmapIndexedNode_assoc(bin, shift, hash_val, key, val, added_leaf, transient_id);
    Py_DECREF(bin);
    return result;
}

static PyObject *HashCollisionNode_find(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *not_found) {
    int idx = HashCollisionNode_find_index(self, key);
    if (idx == -2) return NULL;
    if (idx < 0) {
        Py_INCREF(not_found);
        return not_found;
    }
    PyObject *result = PyList_GET_ITEM(self->array, idx + 1);
    Py_INCREF(result);
    return result;
}

static PyObject *HashCollisionNode_dissoc(HashCollisionNode *self, int shift, Py_hash_t hash_val, PyObject *key, PyObject *removed_leaf, PyObject *transient_id) {
    int idx = HashCollisionNode_find_index(self, key);
    if (idx == -2) return NULL;
    if (idx == -1) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Mark that we found and removed a leaf
    if (removed_leaf && PyList_Append(removed_leaf, Py_True) < 0) return NULL;

    if (self->count == 1) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    Py_ssize_t arr_len = PyList_Size(self->array);
    PyObject *new_array = PyList_New(arr_len - 2);
    if (!new_array) return NULL;

    Py_ssize_t j = 0;
    for (Py_ssize_t i = 0; i < arr_len; i += 2) {
        if (i != idx) {
            PyObject *k = PyList_GET_ITEM(self->array, i);
            PyObject *v = PyList_GET_ITEM(self->array, i + 1);
            Py_INCREF(k);
            Py_INCREF(v);
            PyList_SET_ITEM(new_array, j, k);
            PyList_SET_ITEM(new_array, j + 1, v);
            j += 2;
        }
    }

    HashCollisionNode *node = HashCollisionNode_create(self->hash, self->count - 1, new_array, transient_id);
    Py_DECREF(new_array);
    return (PyObject *)node;
}

// HashCollisionNode iterator
typedef struct {
    PyObject_HEAD
    HashCollisionNode *node;
    Py_ssize_t index;
    int mode;  // ITER_MODE_ITEMS, ITER_MODE_KEYS, or ITER_MODE_VALUES
} HashCollisionNodeIterator;

static PyTypeObject HashCollisionNodeIteratorType;

static void HashCollisionNodeIterator_dealloc(HashCollisionNodeIterator *self) {
    Py_XDECREF(self->node);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *HashCollisionNodeIterator_next(HashCollisionNodeIterator *self) {
    if (self->index >= PyList_Size(self->node->array)) {
        return NULL;
    }

    PyObject *key = PyList_GET_ITEM(self->node->array, self->index);
    PyObject *val = PyList_GET_ITEM(self->node->array, self->index + 1);
    self->index += 2;

    PyObject *result;
    switch (self->mode) {
        case ITER_MODE_KEYS:
            result = key;
            Py_INCREF(result);
            break;
        case ITER_MODE_VALUES:
            result = val;
            Py_INCREF(result);
            break;
        default:  // ITER_MODE_ITEMS
            result = PyTuple_Pack(2, key, val);
            break;
    }
    return result;
}

static PyTypeObject HashCollisionNodeIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.HashCollisionNodeIterator",
    .tp_basicsize = sizeof(HashCollisionNodeIterator),
    .tp_dealloc = (destructor)HashCollisionNodeIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)HashCollisionNodeIterator_next,
};

static PyObject *HashCollisionNode_iter_mode(HashCollisionNode *self, int mode) {
    HashCollisionNodeIterator *it = PyObject_New(HashCollisionNodeIterator, &HashCollisionNodeIteratorType);
    if (!it) return NULL;

    it->node = self;
    Py_INCREF(self);
    it->index = 0;
    it->mode = mode;
    return (PyObject *)it;
}

static PyObject *HashCollisionNode_iter_kv(HashCollisionNode *self) {
    return HashCollisionNode_iter_mode(self, ITER_MODE_ITEMS);
}

// === Map ===
typedef struct Map {
    PyObject_HEAD
    Py_ssize_t cnt;
    PyObject *root;  // BitmapIndexedNode, ArrayNode, or HashCollisionNode
    Py_hash_t hash;
    int hash_computed;
    PyObject *transient_id;
    int initialized;
} Map;

static PyTypeObject MapType;
static Map *EMPTY_MAP = NULL;

static void Map_dealloc(Map *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Map_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Map *self = (Map *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->cnt = 0;
        self->root = NULL;
        self->hash = 0;
        self->hash_computed = 0;
        self->transient_id = NULL;
        self->initialized = 0;
    }
    return (PyObject *)self;
}

static Map *Map_create(Py_ssize_t cnt, PyObject *root, PyObject *transient_id) {
    Map *m = (Map *)MapType.tp_alloc(&MapType, 0);
    if (!m) return NULL;

    m->cnt = cnt;
    m->root = root;
    Py_XINCREF(root);
    m->hash = 0;
    m->hash_computed = 0;
    m->transient_id = transient_id;
    Py_XINCREF(transient_id);
    m->initialized = 1;

    return m;
}

static Py_ssize_t Map_length(Map *self) {
    return self->cnt;
}

static PyObject *Map_get(Map *self, PyObject *args) {
    PyObject *key;
    PyObject *default_val = Py_None;

    if (!PyArg_ParseTuple(args, "O|O", &key, &default_val)) {
        return NULL;
    }

    if (self->root == NULL) {
        Py_INCREF(default_val);
        return default_val;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, default_val);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_find((ArrayNode *)self->root, 0, h, key, default_val);
    } else {
        return HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, default_val);
    }
}

static PyObject *Map_getitem(Map *self, PyObject *key) {
    if (self->root == NULL) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *result;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        result = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        result = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        result = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!result) return NULL;

    if (result == _MISSING) {
        Py_DECREF(result);
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    return result;
}

static int Map_contains(Map *self, PyObject *key) {
    if (self->root == NULL) {
        return 0;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return -1;

    PyObject *result;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        result = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        result = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        result = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!result) return -1;

    int found = (result != _MISSING);
    Py_DECREF(result);
    return found;
}

static PyObject *Map_assoc(Map *self, PyObject *args) {
    PyObject *key, *val;

    if (!PyArg_ParseTuple(args, "OO", &key, &val)) {
        return NULL;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *added_leaf = PyList_New(0);
    if (!added_leaf) return NULL;

    PyObject *root = self->root ? self->root : (PyObject *)EMPTY_BIN;
    PyObject *new_root;

    if (PyObject_TypeCheck(root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_assoc((BitmapIndexedNode *)root, 0, h, key, val, added_leaf, self->transient_id);
    } else if (PyObject_TypeCheck(root, &ArrayNodeType)) {
        new_root = ArrayNode_assoc((ArrayNode *)root, 0, h, key, val, added_leaf, self->transient_id);
    } else {
        new_root = HashCollisionNode_assoc((HashCollisionNode *)root, 0, h, key, val, added_leaf, self->transient_id);
    }

    if (!new_root) {
        Py_DECREF(added_leaf);
        return NULL;
    }

    if (new_root == self->root) {
        Py_DECREF(new_root);
        Py_DECREF(added_leaf);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_ssize_t new_cnt = self->cnt + PyList_Size(added_leaf);
    Py_DECREF(added_leaf);

    Map *result = Map_create(new_cnt, new_root, self->transient_id);
    Py_DECREF(new_root);
    return (PyObject *)result;
}

static PyObject *Map_dissoc(Map *self, PyObject *key) {
    if (self->root == NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *new_root;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_dissoc((BitmapIndexedNode *)self->root, 0, h, key, NULL, self->transient_id);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        new_root = ArrayNode_dissoc((ArrayNode *)self->root, 0, h, key, NULL, self->transient_id);
    } else {
        new_root = HashCollisionNode_dissoc((HashCollisionNode *)self->root, 0, h, key, NULL, self->transient_id);
    }

    if (!new_root) return NULL;

    if (new_root == self->root) {
        Py_DECREF(new_root);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_ssize_t new_cnt = (new_root != Py_None) ? self->cnt - 1 : 0;
    if (new_root == Py_None) {
        Py_DECREF(new_root);
        new_root = NULL;
    }

    Map *result = Map_create(new_cnt, new_root, self->transient_id);
    Py_XDECREF(new_root);
    return (PyObject *)result;
}

static PyObject *Map_iter(Map *self) {
    if (self->root == NULL) {
        return PyObject_GetIter(PyList_New(0));
    }

    // Get a key-only iterator directly (no tuple allocation)
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_mode((BitmapIndexedNode *)self->root, ITER_MODE_KEYS);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_mode((ArrayNode *)self->root, ITER_MODE_KEYS);
    } else {
        return HashCollisionNode_iter_mode((HashCollisionNode *)self->root, ITER_MODE_KEYS);
    }
}

static PyObject *Map_items(Map *self, PyObject *Py_UNUSED(ignored)) {
    if (self->root == NULL) {
        return PyObject_GetIter(PyList_New(0));
    }

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_kv((BitmapIndexedNode *)self->root);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_kv((ArrayNode *)self->root);
    } else {
        return HashCollisionNode_iter_kv((HashCollisionNode *)self->root);
    }
}

static PyObject *Map_keys(Map *self, PyObject *Py_UNUSED(ignored)) {
    return Map_iter(self);
}

static PyObject *Map_values(Map *self, PyObject *Py_UNUSED(ignored)) {
    if (self->root == NULL) {
        return PyObject_GetIter(PyList_New(0));
    }

    // Get a value-only iterator directly (no tuple allocation)
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_mode((BitmapIndexedNode *)self->root, ITER_MODE_VALUES);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_mode((ArrayNode *)self->root, ITER_MODE_VALUES);
    } else {
        return HashCollisionNode_iter_mode((HashCollisionNode *)self->root, ITER_MODE_VALUES);
    }
}

static Py_hash_t Map_hash(Map *self) {
    if (self->hash_computed) {
        return self->hash;
    }

    Py_hash_t h = 0;
    PyObject *items_iter = Map_items(self, NULL);
    if (!items_iter) return -1;

    PyObject *pair;
    while ((pair = PyIter_Next(items_iter)) != NULL) {
        PyObject *key = PyTuple_GET_ITEM(pair, 0);
        PyObject *val = PyTuple_GET_ITEM(pair, 1);

        Py_hash_t kh = PyObject_Hash(key);
        Py_hash_t vh = PyObject_Hash(val);
        Py_DECREF(pair);

        if ((kh == -1 || vh == -1) && PyErr_Occurred()) {
            Py_DECREF(items_iter);
            return -1;
        }

        h += kh ^ vh;
    }
    Py_DECREF(items_iter);

    if (PyErr_Occurred()) return -1;

    self->hash = h;
    self->hash_computed = 1;
    return h;
}

static PyObject *Map_richcompare(Map *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self == (Map *)other) {
        return PyBool_FromLong(op == Py_EQ);
    }

    if (!PyObject_TypeCheck(other, &MapType)) {
        return PyBool_FromLong(op == Py_NE);
    }

    Map *o = (Map *)other;
    if (self->cnt != o->cnt) {
        return PyBool_FromLong(op == Py_NE);
    }

    PyObject *items_iter = Map_items(self, NULL);
    if (!items_iter) return NULL;

    PyObject *pair;
    while ((pair = PyIter_Next(items_iter)) != NULL) {
        PyObject *key = PyTuple_GET_ITEM(pair, 0);
        PyObject *val = PyTuple_GET_ITEM(pair, 1);

        Py_hash_t h = PyObject_Hash(key);
        if (h == -1 && PyErr_Occurred()) {
            Py_DECREF(pair);
            Py_DECREF(items_iter);
            return NULL;
        }

        PyObject *other_val;
        if (o->root == NULL) {
            other_val = _MISSING;
            Py_INCREF(_MISSING);
        } else if (PyObject_TypeCheck(o->root, &BitmapIndexedNodeType)) {
            other_val = BitmapIndexedNode_find((BitmapIndexedNode *)o->root, 0, h, key, _MISSING);
        } else if (PyObject_TypeCheck(o->root, &ArrayNodeType)) {
            other_val = ArrayNode_find((ArrayNode *)o->root, 0, h, key, _MISSING);
        } else {
            other_val = HashCollisionNode_find((HashCollisionNode *)o->root, 0, h, key, _MISSING);
        }

        if (!other_val) {
            Py_DECREF(pair);
            Py_DECREF(items_iter);
            return NULL;
        }

        if (other_val == _MISSING) {
            Py_DECREF(other_val);
            Py_DECREF(pair);
            Py_DECREF(items_iter);
            return PyBool_FromLong(op == Py_NE);
        }

        int eq = PyObject_RichCompareBool(val, other_val, Py_EQ);
        Py_DECREF(other_val);
        Py_DECREF(pair);

        if (eq < 0) {
            Py_DECREF(items_iter);
            return NULL;
        }
        if (!eq) {
            Py_DECREF(items_iter);
            return PyBool_FromLong(op == Py_NE);
        }
    }
    Py_DECREF(items_iter);

    if (PyErr_Occurred()) return NULL;

    return PyBool_FromLong(op == Py_EQ);
}

static PyObject *Map_repr(Map *self) {
    PyObject *parts = PyList_New(0);
    if (!parts) return NULL;

    PyObject *items_iter = Map_items(self, NULL);
    if (!items_iter) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *pair;
    while ((pair = PyIter_Next(items_iter)) != NULL) {
        PyObject *key = PyTuple_GET_ITEM(pair, 0);
        PyObject *val = PyTuple_GET_ITEM(pair, 1);

        PyObject *key_repr = PyObject_Repr(key);
        PyObject *val_repr = PyObject_Repr(val);
        Py_DECREF(pair);

        if (!key_repr || !val_repr) {
            Py_XDECREF(key_repr);
            Py_XDECREF(val_repr);
            Py_DECREF(items_iter);
            Py_DECREF(parts);
            return NULL;
        }

        PyObject *part = PyUnicode_FromFormat("%U %U", key_repr, val_repr);
        Py_DECREF(key_repr);
        Py_DECREF(val_repr);

        if (!part) {
            Py_DECREF(items_iter);
            Py_DECREF(parts);
            return NULL;
        }

        if (PyList_Append(parts, part) < 0) {
            Py_DECREF(part);
            Py_DECREF(items_iter);
            Py_DECREF(parts);
            return NULL;
        }
        Py_DECREF(part);
    }
    Py_DECREF(items_iter);

    if (PyErr_Occurred()) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *space = PyUnicode_FromString(" ");
    if (!space) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *joined = PyUnicode_Join(space, parts);
    Py_DECREF(space);
    Py_DECREF(parts);
    if (!joined) return NULL;

    PyObject *result = PyUnicode_FromFormat("{%U}", joined);
    Py_DECREF(joined);
    return result;
}

// TransientMap forward declaration
static PyTypeObject TransientMapType;

static PyObject *Map_transient(Map *self, PyObject *Py_UNUSED(ignored));
static PyObject *TransientMap_assoc_mut_impl(TransientMap *self, PyObject *key, PyObject *val);
static PyObject *TransientMap_persistent(TransientMap *self, PyObject *Py_UNUSED(ignored));

static PyObject *Map_to_seq(Map *self, PyObject *Py_UNUSED(ignored)) {
    if (self->cnt == 0) {
        Py_RETURN_NONE;
    }

    // Build Cons list of [k v] vectors
    PyObject *items_iter = Map_items(self, NULL);
    if (!items_iter) return NULL;

    PyObject *pairs = PyList_New(0);
    if (!pairs) {
        Py_DECREF(items_iter);
        return NULL;
    }

    PyObject *pair;
    while ((pair = PyIter_Next(items_iter)) != NULL) {
        PyObject *key = PyTuple_GET_ITEM(pair, 0);
        PyObject *val = PyTuple_GET_ITEM(pair, 1);

        // Create Vector [key, val]
        Vector *kv = (Vector *)VectorType.tp_alloc(&VectorType, 0);
        if (!kv) {
            Py_DECREF(pair);
            Py_DECREF(items_iter);
            Py_DECREF(pairs);
            return NULL;
        }
        kv->cnt = 2;
        kv->shift = BITS;
        kv->root = EMPTY_NODE;
        Py_INCREF(EMPTY_NODE);
        kv->tail = PyTuple_Pack(2, key, val);
        kv->hash = 0;
        kv->hash_computed = 0;
        kv->transient_id = NULL;

        Py_DECREF(pair);

        if (!kv->tail) {
            Py_DECREF(kv);
            Py_DECREF(items_iter);
            Py_DECREF(pairs);
            return NULL;
        }

        if (PyList_Append(pairs, (PyObject *)kv) < 0) {
            Py_DECREF(kv);
            Py_DECREF(items_iter);
            Py_DECREF(pairs);
            return NULL;
        }
        Py_DECREF(kv);
    }
    Py_DECREF(items_iter);

    if (PyErr_Occurred()) {
        Py_DECREF(pairs);
        return NULL;
    }

    // Build Cons list in reverse
    Cons *result = NULL;
    for (Py_ssize_t i = PyList_Size(pairs) - 1; i >= 0; i--) {
        PyObject *item = PyList_GET_ITEM(pairs, i);
        Cons *new_cons = (Cons *)ConsType.tp_alloc(&ConsType, 0);
        if (!new_cons) {
            Py_XDECREF(result);
            Py_DECREF(pairs);
            return NULL;
        }

        Py_INCREF(item);
        new_cons->first = item;
        new_cons->rest = result ? (PyObject *)result : Py_None;
        Py_INCREF(new_cons->rest);
        new_cons->hash = 0;
        new_cons->hash_computed = 0;

        result = new_cons;
    }

    Py_DECREF(pairs);
    return (PyObject *)result;
}

// Map merge operation (|)
static PyObject *Map_or(PyObject *left, PyObject *right) {
    if (!PyObject_TypeCheck(left, &MapType)) {
        // Reflected merge: Mapping | Map. Build a persistent map from the
        // left mapping first, then apply the right map so right-hand values
        // retain normal dict-union precedence.
        if (!PyObject_TypeCheck(right, &MapType)) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        int has_items = PyDict_Check(left) ? 1 : PyObject_HasAttrString(left, "items");
        if (has_items < 0) return NULL;
        if (!has_items) Py_RETURN_NOTIMPLEMENTED;

        PyObject *base = Map_or((PyObject *)EMPTY_MAP, left);
        if (!base) return NULL;
        if (base == Py_NotImplemented) {
            Py_DECREF(base);
            Py_RETURN_NOTIMPLEMENTED;
        }

        PyObject *result = Map_or(base, right);
        Py_DECREF(base);
        return result;
    }
    Map *self = (Map *)left;

    // If right is empty, return self
    if (PyObject_TypeCheck(right, &MapType)) {
        Map *other = (Map *)right;
        if (other->cnt == 0) {
            Py_INCREF(self);
            return (PyObject *)self;
        }
        if (self->cnt == 0) {
            Py_INCREF(other);
            return (PyObject *)other;
        }
    }

    // Create transient from self via direct C call
    TransientMap *t = (TransientMap *)Map_transient(self, NULL);
    if (!t) return NULL;

    // Get items iterator for right
    PyObject *items_iter;

    if (PyObject_TypeCheck(right, &MapType)) {
        // Fast internal iterator for Map
        items_iter = Map_items((Map *)right, NULL);
    } else if (PyDict_Check(right)) {
        // Fast path for Python dicts
        items_iter = PyObject_CallMethod(right, "items", NULL);
        if (items_iter) {
            PyObject *temp = PyObject_GetIter(items_iter);
            Py_DECREF(items_iter);
            items_iter = temp;
        }
    } else {
        // Dict-style union accepts mappings, not arbitrary iterables of pairs.
        int has_items = PyObject_HasAttrString(right, "items");
        if (has_items < 0) {
            Py_DECREF(t);
            return NULL;
        }
        if (!has_items) {
            Py_DECREF(t);
            Py_RETURN_NOTIMPLEMENTED;
        }

        PyObject *items_method = PyObject_GetAttrString(right, "items");
        if (!items_method) {
            Py_DECREF(t);
            return NULL;
        }
        items_iter = PyObject_CallObject(items_method, NULL);
        Py_DECREF(items_method);
        if (items_iter) {
            PyObject *temp = PyObject_GetIter(items_iter);
            Py_DECREF(items_iter);
            items_iter = temp;
        }
    }

    if (!items_iter) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_DECREF(t);
            Py_RETURN_NOTIMPLEMENTED;
        }
        Py_DECREF(t);
        return NULL;
    }

    PyObject *item;
    while ((item = PyIter_Next(items_iter)) != NULL) {
        PyObject *key, *val;

        // Handle tuples, lists, Vectors, and other sequences
        if (PyTuple_Check(item) && PyTuple_GET_SIZE(item) == 2) {
            key = PyTuple_GET_ITEM(item, 0);
            val = PyTuple_GET_ITEM(item, 1);
        } else if (PyList_Check(item) && PyList_GET_SIZE(item) == 2) {
            key = PyList_GET_ITEM(item, 0);
            val = PyList_GET_ITEM(item, 1);
        } else if (PyObject_TypeCheck(item, &VectorType)) {
            // Handle our Vector type directly using internal C API
            Vector *vec = (Vector *)item;
            if (vec->cnt != 2) {
                PyErr_SetString(PyExc_ValueError, "Map merge requires (key, value) pairs");
                Py_DECREF(item);
                Py_DECREF(items_iter);
                Py_DECREF(t);
                return NULL;
            }
            key = Vector_nth_impl(vec, 0, NULL);
            val = Vector_nth_impl(vec, 1, NULL);
            if (!key || !val) {
                Py_XDECREF(key);
                Py_XDECREF(val);
                Py_DECREF(item);
                Py_DECREF(items_iter);
                Py_DECREF(t);
                return NULL;
            }
            PyObject *res = TransientMap_assoc_mut_impl(t, key, val);
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(item);
            if (!res) {
                Py_DECREF(items_iter);
                Py_DECREF(t);
                return NULL;
            }
            Py_DECREF(res);
            continue;
        } else if (PySequence_Check(item) && PySequence_Size(item) == 2) {
            key = PySequence_GetItem(item, 0);
            val = PySequence_GetItem(item, 1);
            if (!key || !val) {
                Py_XDECREF(key);
                Py_XDECREF(val);
                Py_DECREF(item);
                Py_DECREF(items_iter);
                Py_DECREF(t);
                return NULL;
            }
            // Use impl and then decref the borrowed refs
            PyObject *res = TransientMap_assoc_mut_impl(t, key, val);
            Py_DECREF(key);
            Py_DECREF(val);
            Py_DECREF(item);
            if (!res) {
                Py_DECREF(items_iter);
                Py_DECREF(t);
                return NULL;
            }
            Py_DECREF(res);
            continue;
        } else {
            PyErr_SetString(PyExc_ValueError, "Map merge requires (key, value) pairs");
            Py_DECREF(item);
            Py_DECREF(items_iter);
            Py_DECREF(t);
            return NULL;
        }

        // OPTIMIZATION: Call internal C function directly
        PyObject *res = TransientMap_assoc_mut_impl(t, key, val);
        Py_DECREF(item);
        if (!res) {
            Py_DECREF(items_iter);
            Py_DECREF(t);
            return NULL;
        }
        Py_DECREF(res);
    }

    Py_DECREF(items_iter);

    if (PyErr_Occurred()) {
        Py_DECREF(t);
        return NULL;
    }

    PyObject *result = TransientMap_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}

static int Map_replace_contents(Map *self, PyObject *source) {
    PyObject *merged = Map_or((PyObject *)self, source);
    if (!merged) return -1;
    if (merged == Py_NotImplemented) {
        Py_DECREF(merged);
        PyErr_SetString(
            PyExc_TypeError,
            "Map argument must be a mapping or an iterable of (key, value) pairs"
        );
        return -1;
    }

    Map *new_map = (Map *)merged;
    PyObject *new_root = new_map->root;
    PyObject *new_transient_id = new_map->transient_id;
    Py_XINCREF(new_root);
    Py_XINCREF(new_transient_id);

    Py_XDECREF(self->root);
    Py_XDECREF(self->transient_id);
    self->cnt = new_map->cnt;
    self->root = new_root;
    self->hash = new_map->hash;
    self->hash_computed = new_map->hash_computed;
    self->transient_id = new_transient_id;

    Py_DECREF(merged);
    return 0;
}

static int Map_init(Map *self, PyObject *args, PyObject *kwds) {
    if (self->initialized) {
        PyErr_SetString(PyExc_TypeError, "Map values cannot be reinitialized");
        return -1;
    }

    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs > 1) {
        PyErr_Format(
            PyExc_TypeError,
            "Map expected at most 1 argument, got %zd",
            nargs
        );
        return -1;
    }

    if (nargs == 1) {
        // Normalize through dict so construction accepts either a mapping or
        // an iterable of pairs while the | operator remains mapping-only.
        PyObject *source = PyTuple_GET_ITEM(args, 0);
        PyObject *mapping = PyObject_CallFunctionObjArgs(
            (PyObject *)&PyDict_Type,
            source,
            NULL
        );
        if (!mapping) return -1;

        int status = Map_replace_contents(self, mapping);
        Py_DECREF(mapping);
        if (status < 0) return -1;
    }

    if (kwds && PyDict_GET_SIZE(kwds) > 0 && Map_replace_contents(self, kwds) < 0) {
        return -1;
    }

    self->initialized = 1;
    return 0;
}

static PyNumberMethods Map_as_number = {
    .nb_or = Map_or,
};

/* Map.copy() - returns self since Map is immutable */
static PyObject *Map_copy(Map *self, PyObject *Py_UNUSED(ignored)) {
    Py_INCREF(self);
    return (PyObject *)self;
}

// Forward declaration for hash_map function used by Map_reduce
static PyObject *pds_hash_map(PyObject *self, PyObject *args);

static PyObject *Map_reduce(Map *self, PyObject *Py_UNUSED(ignored)) {
    // Build a tuple of (k1, v1, k2, v2, ...) for hash_map reconstructor
    PyObject *args = PyTuple_New(self->cnt * 2);
    if (args == NULL) {
        return NULL;
    }

    // Iterate over items and flatten into the args tuple
    PyObject *items_iter = Map_items(self, NULL);
    if (items_iter == NULL) {
        Py_DECREF(args);
        return NULL;
    }

    Py_ssize_t i = 0;
    PyObject *item;
    while ((item = PyIter_Next(items_iter)) != NULL) {
        PyObject *key = PyTuple_GET_ITEM(item, 0);
        PyObject *val = PyTuple_GET_ITEM(item, 1);
        Py_INCREF(key);
        Py_INCREF(val);
        PyTuple_SET_ITEM(args, i * 2, key);
        PyTuple_SET_ITEM(args, i * 2 + 1, val);
        Py_DECREF(item);
        i++;
    }
    Py_DECREF(items_iter);

    if (PyErr_Occurred()) {
        Py_DECREF(args);
        return NULL;
    }

    // Get the hash_map function from the pds module
    PyObject *pds_module = PyImport_ImportModule("spork_pds");
    if (pds_module == NULL) {
        Py_DECREF(args);
        return NULL;
    }

    PyObject *hash_map_func = PyObject_GetAttrString(pds_module, "hash_map");
    Py_DECREF(pds_module);
    if (hash_map_func == NULL) {
        Py_DECREF(args);
        return NULL;
    }

    // Return (hash_map, args_tuple) - pickle will call hash_map(*args_tuple)
    PyObject *result = PyTuple_Pack(2, hash_map_func, args);
    Py_DECREF(hash_map_func);
    Py_DECREF(args);
    return result;
}

static PyMethodDef Map_methods[] = {
    {"get", (PyCFunction)Map_get, METH_VARARGS, "Get value for key"},
    {"assoc", (PyCFunction)Map_assoc, METH_VARARGS, "Set key to value"},
    {"dissoc", (PyCFunction)Map_dissoc, METH_O, "Remove key"},
    {"items", (PyCFunction)Map_items, METH_NOARGS, "Iterate over key-value pairs"},
    {"keys", (PyCFunction)Map_keys, METH_NOARGS, "Iterate over keys"},
    {"values", (PyCFunction)Map_values, METH_NOARGS, "Iterate over values"},
    {"transient", (PyCFunction)Map_transient, METH_NOARGS, "Get transient version"},
    {"to_seq", (PyCFunction)Map_to_seq, METH_NOARGS, "Convert to Cons sequence"},
    {"copy", (PyCFunction)Map_copy, METH_NOARGS, "Return self (immutable maps don't need copying)"},
    {"__reduce__", (PyCFunction)Map_reduce, METH_NOARGS, "Pickle support"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations (e.g., Map[str, int])"},
    {NULL}
};

static PySequenceMethods Map_as_sequence = {
    .sq_contains = (objobjproc)Map_contains,
};

static PyMappingMethods Map_as_mapping = {
    .mp_length = (lenfunc)Map_length,
    .mp_subscript = (binaryfunc)Map_getitem,
};

static PyTypeObject MapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.Map",
    .tp_doc = "Map(mapping_or_pairs=(), **kwargs) -> persistent hash map using a HAMT",
    .tp_basicsize = sizeof(Map),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Map_dealloc,
    .tp_repr = (reprfunc)Map_repr,
    .tp_as_number = &Map_as_number,
    .tp_as_sequence = &Map_as_sequence,
    .tp_as_mapping = &Map_as_mapping,
    .tp_hash = (hashfunc)Map_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_richcompare = (richcmpfunc)Map_richcompare,
    .tp_iter = (getiterfunc)Map_iter,
    .tp_methods = Map_methods,
    .tp_init = (initproc)Map_init,
    .tp_new = Map_new,
};

// === TransientMap ===
typedef struct TransientMap {
    PyObject_HEAD
    Py_ssize_t cnt;
    PyObject *root;
    PyObject *id;
} TransientMap;

static void TransientMap_dealloc(TransientMap *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Map_transient(Map *self, PyObject *Py_UNUSED(ignored)) {
    TransientMap *t = PyObject_New(TransientMap, &TransientMapType);
    if (!t) return NULL;

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        return NULL;
    }

    t->cnt = self->cnt;
    if (self->root != NULL) {
        if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
            t->root = (PyObject *)BitmapIndexedNode_ensure_editable((BitmapIndexedNode *)self->root, t->id);
        } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
            t->root = (PyObject *)ArrayNode_ensure_editable((ArrayNode *)self->root, t->id);
        } else {
            t->root = (PyObject *)HashCollisionNode_ensure_editable((HashCollisionNode *)self->root, t->id);
        }
        if (!t->root) {
            Py_DECREF(t);
            return NULL;
        }
    } else {
        t->root = NULL;
    }

    return (PyObject *)t;
}

static void TransientMap_ensure_editable(TransientMap *self) {
    if (self->id == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Transient used after persistent() call");
    }
}

// Internal C API - no argument parsing overhead
static PyObject *TransientMap_assoc_mut_impl(TransientMap *self, PyObject *key, PyObject *val) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *added_leaf = PyList_New(0);
    if (!added_leaf) return NULL;

    PyObject *root = self->root ? self->root : (PyObject *)EMPTY_BIN;
    PyObject *new_root;

    if (PyObject_TypeCheck(root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_assoc((BitmapIndexedNode *)root, 0, h, key, val, added_leaf, self->id);
    } else if (PyObject_TypeCheck(root, &ArrayNodeType)) {
        new_root = ArrayNode_assoc((ArrayNode *)root, 0, h, key, val, added_leaf, self->id);
    } else {
        new_root = HashCollisionNode_assoc((HashCollisionNode *)root, 0, h, key, val, added_leaf, self->id);
    }

    if (!new_root) {
        Py_DECREF(added_leaf);
        return NULL;
    }

    if (new_root != self->root) {
        Py_XDECREF(self->root);
        self->root = new_root;
    } else {
        Py_DECREF(new_root);
    }

    self->cnt += PyList_Size(added_leaf);
    Py_DECREF(added_leaf);

    Py_INCREF(self);
    return (PyObject *)self;
}

// Python wrapper - parses arguments then calls impl
static PyObject *TransientMap_assoc_mut(TransientMap *self, PyObject *args) {
    PyObject *key, *val;
    if (!PyArg_ParseTuple(args, "OO", &key, &val)) {
        return NULL;
    }
    return TransientMap_assoc_mut_impl(self, key, val);
}

static PyObject *TransientMap_dissoc_mut(TransientMap *self, PyObject *key) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *removed_leaf = PyList_New(0);
    if (!removed_leaf) return NULL;

    PyObject *new_root;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_dissoc((BitmapIndexedNode *)self->root, 0, h, key, removed_leaf, self->id);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        new_root = ArrayNode_dissoc((ArrayNode *)self->root, 0, h, key, removed_leaf, self->id);
    } else {
        new_root = HashCollisionNode_dissoc((HashCollisionNode *)self->root, 0, h, key, removed_leaf, self->id);
    }

    if (!new_root && PyErr_Occurred()) {
        Py_DECREF(removed_leaf);
        return NULL;
    }

    // Check if a key was actually removed using the removed_leaf flag
    int key_was_removed = PyList_Size(removed_leaf) > 0;
    Py_DECREF(removed_leaf);

    if (new_root != self->root) {
        // Root changed
        if (new_root == Py_None) {
            // Map is now empty
            Py_DECREF(new_root);
            Py_XDECREF(self->root);
            self->root = NULL;
            self->cnt = 0;
        } else {
            Py_XDECREF(self->root);
            self->root = new_root;
            if (key_was_removed) {
                self->cnt--;
            }
        }
    } else if (new_root) {
        // Root unchanged, but key might have been removed via in-place mutation
        Py_DECREF(new_root);
        if (key_was_removed) {
            self->cnt--;
        }
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientMap_persistent(TransientMap *self, PyObject *Py_UNUSED(ignored)) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_CLEAR(self->id);

    Map *result = Map_create(self->cnt, self->root, NULL);
    return (PyObject *)result;
}

// === TransientMap MutableMapping Protocol ===

static Py_ssize_t TransientMap_length(TransientMap *self) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return -1;
    return self->cnt;
}

static PyObject *TransientMap_getitem(TransientMap *self, PyObject *key) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *result;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        result = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        result = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        result = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!result) return NULL;

    if (result == _MISSING) {
        Py_DECREF(result);
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    return result;
}

static int TransientMap_ass_subscript(TransientMap *self, PyObject *key, PyObject *val) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return -1;

    if (val != NULL) {
        // Set: t[k] = v
        PyObject *result = TransientMap_assoc_mut_impl(self, key, val);
        if (!result) return -1;
        Py_DECREF(result);
        return 0;
    } else {
        // Delete: del t[k]
        // First check if key exists (Python semantics require KeyError if missing)
        Py_hash_t h = PyObject_Hash(key);
        if (h == -1 && PyErr_Occurred()) return -1;

        if (self->root != NULL) {
            PyObject *found;
            if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
                found = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
            } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
                found = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
            } else {
                found = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
            }

            if (!found) return -1;

            if (found == _MISSING) {
                Py_DECREF(found);
                PyErr_SetObject(PyExc_KeyError, key);
                return -1;
            }
            Py_DECREF(found);
        } else {
            PyErr_SetObject(PyExc_KeyError, key);
            return -1;
        }

        PyObject *result = TransientMap_dissoc_mut(self, key);
        if (!result) return -1;
        Py_DECREF(result);
        return 0;
    }
}

static int TransientMap_contains(TransientMap *self, PyObject *key) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return -1;

    if (self->root == NULL) {
        return 0;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return -1;

    PyObject *result;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        result = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        result = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        result = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!result) return -1;

    int found = (result != _MISSING);
    Py_DECREF(result);
    return found;
}

static PyObject *TransientMap_iter(TransientMap *self) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        // Return empty iterator
        return PyObject_GetIter(PyTuple_New(0));
    }

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_mode((BitmapIndexedNode *)self->root, ITER_MODE_KEYS);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_mode((ArrayNode *)self->root, ITER_MODE_KEYS);
    } else {
        return HashCollisionNode_iter_mode((HashCollisionNode *)self->root, ITER_MODE_KEYS);
    }
}

static PyObject *TransientMap_keys(TransientMap *self, PyObject *Py_UNUSED(ignored)) {
    return TransientMap_iter(self);
}

static PyObject *TransientMap_values(TransientMap *self, PyObject *Py_UNUSED(ignored)) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        return PyObject_GetIter(PyTuple_New(0));
    }

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_mode((BitmapIndexedNode *)self->root, ITER_MODE_VALUES);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_mode((ArrayNode *)self->root, ITER_MODE_VALUES);
    } else {
        return HashCollisionNode_iter_mode((HashCollisionNode *)self->root, ITER_MODE_VALUES);
    }
}

static PyObject *TransientMap_items(TransientMap *self, PyObject *Py_UNUSED(ignored)) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        return PyObject_GetIter(PyTuple_New(0));
    }

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_mode((BitmapIndexedNode *)self->root, ITER_MODE_ITEMS);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_mode((ArrayNode *)self->root, ITER_MODE_ITEMS);
    } else {
        return HashCollisionNode_iter_mode((HashCollisionNode *)self->root, ITER_MODE_ITEMS);
    }
}

static PyObject *TransientMap_get(TransientMap *self, PyObject *args) {
    PyObject *key;
    PyObject *default_val = Py_None;

    if (!PyArg_ParseTuple(args, "O|O", &key, &default_val)) {
        return NULL;
    }

    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        Py_INCREF(default_val);
        return default_val;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, default_val);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_find((ArrayNode *)self->root, 0, h, key, default_val);
    } else {
        return HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, default_val);
    }
}

static PyMethodDef TransientMap_methods[] = {
    {"assoc_mut", (PyCFunction)TransientMap_assoc_mut, METH_VARARGS, "Mutably set key to value"},
    {"dissoc_mut", (PyCFunction)TransientMap_dissoc_mut, METH_O, "Mutably remove key"},
    {"persistent", (PyCFunction)TransientMap_persistent, METH_NOARGS, "Return persistent map"},
    {"get", (PyCFunction)TransientMap_get, METH_VARARGS, "Get value for key with optional default"},
    {"keys", (PyCFunction)TransientMap_keys, METH_NOARGS, "Iterate over keys"},
    {"values", (PyCFunction)TransientMap_values, METH_NOARGS, "Iterate over values"},
    {"items", (PyCFunction)TransientMap_items, METH_NOARGS, "Iterate over key-value pairs"},
    {NULL}
};

static PySequenceMethods TransientMap_as_sequence = {
    .sq_contains = (objobjproc)TransientMap_contains,
};

static PyMappingMethods TransientMap_as_mapping = {
    .mp_length = (lenfunc)TransientMap_length,
    .mp_subscript = (binaryfunc)TransientMap_getitem,
    .mp_ass_subscript = (objobjargproc)TransientMap_ass_subscript,
};

static PyTypeObject TransientMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientMap",
    .tp_doc = "Transient map for batch operations",
    .tp_basicsize = sizeof(TransientMap),
    .tp_dealloc = (destructor)TransientMap_dealloc,
    .tp_as_sequence = &TransientMap_as_sequence,
    .tp_as_mapping = &TransientMap_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = (getiterfunc)TransientMap_iter,
    .tp_methods = TransientMap_methods,
};

// === Set ===
// Immutable hash set using HAMT (same structure as Map, but values are always Py_None)

struct Set {
    PyObject_HEAD
    Py_ssize_t cnt;
    PyObject *root;  // BitmapIndexedNode, ArrayNode, or HashCollisionNode
    Py_hash_t hash;
    int hash_computed;
    PyObject *transient_id;
    int initialized;
};

static PyTypeObject SetType;
static Set *EMPTY_SET = NULL;

static void Set_dealloc(Set *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Set_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Set *self = (Set *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->cnt = 0;
        self->root = NULL;
        self->hash = 0;
        self->hash_computed = 0;
        self->transient_id = NULL;
        self->initialized = 0;
    }
    return (PyObject *)self;
}

// Forward declarations for Set_init
static PyObject *Set_conj(Set *self, PyObject *val);

static int Set_init(Set *self, PyObject *args, PyObject *kwds) {
    if (self->initialized) {
        PyErr_SetString(PyExc_TypeError, "Set values cannot be reinitialized");
        return -1;
    }

    Py_ssize_t nargs = PyTuple_GET_SIZE(args);

    if (kwds && PyDict_GET_SIZE(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "Set takes no keyword arguments");
        return -1;
    }
    if (nargs > 1) {
        PyErr_Format(
            PyExc_TypeError,
            "Set expected at most 1 argument, got %zd",
            nargs
        );
        return -1;
    }
    if (nargs == 0) {
        self->initialized = 1;
        return 0;
    }

    PyObject *iter = PyObject_GetIter(PyTuple_GET_ITEM(args, 0));
    if (!iter) return -1;

    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        PyObject *new_set = Set_conj(self, item);
        Py_DECREF(item);
        if (!new_set) {
            Py_DECREF(iter);
            return -1;
        }

        // Update self from new_set while retaining its structurally shared root.
        Set *ns = (Set *)new_set;
        PyObject *new_root = ns->root;
        Py_XINCREF(new_root);
        Py_XDECREF(self->root);
        self->cnt = ns->cnt;
        self->root = new_root;
        self->hash = 0;
        self->hash_computed = 0;
        Py_DECREF(new_set);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) return -1;
    self->initialized = 1;
    return 0;
}

static Set *Set_create(Py_ssize_t cnt, PyObject *root, PyObject *transient_id) {
    Set *s = (Set *)SetType.tp_alloc(&SetType, 0);
    if (!s) return NULL;

    s->cnt = cnt;
    s->root = root;
    Py_XINCREF(root);
    s->hash = 0;
    s->hash_computed = 0;
    s->transient_id = transient_id;
    Py_XINCREF(transient_id);
    s->initialized = 1;

    return s;
}

static Py_ssize_t Set_length(Set *self) {
    return self->cnt;
}

static int Set_contains(Set *self, PyObject *key) {
    if (self->root == NULL) {
        return 0;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return -1;

    PyObject *result;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        result = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        result = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        result = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!result) return -1;

    int found = (result != _MISSING);
    Py_DECREF(result);
    return found;
}

static PyObject *Set_conj(Set *self, PyObject *key) {
    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *added_leaf = PyList_New(0);
    if (!added_leaf) return NULL;

    PyObject *root = self->root ? self->root : (PyObject *)EMPTY_BIN;
    PyObject *new_root;

    if (PyObject_TypeCheck(root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_assoc((BitmapIndexedNode *)root, 0, h, key, Py_None, added_leaf, self->transient_id);
    } else if (PyObject_TypeCheck(root, &ArrayNodeType)) {
        new_root = ArrayNode_assoc((ArrayNode *)root, 0, h, key, Py_None, added_leaf, self->transient_id);
    } else {
        new_root = HashCollisionNode_assoc((HashCollisionNode *)root, 0, h, key, Py_None, added_leaf, self->transient_id);
    }

    if (!new_root) {
        Py_DECREF(added_leaf);
        return NULL;
    }

    if (new_root == self->root) {
        Py_DECREF(new_root);
        Py_DECREF(added_leaf);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_ssize_t new_cnt = self->cnt + PyList_Size(added_leaf);
    Py_DECREF(added_leaf);

    Set *result = Set_create(new_cnt, new_root, self->transient_id);
    Py_DECREF(new_root);
    return (PyObject *)result;
}

static PyObject *Set_disj(Set *self, PyObject *key) {
    if (self->root == NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *new_root;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_dissoc((BitmapIndexedNode *)self->root, 0, h, key, NULL, self->transient_id);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        new_root = ArrayNode_dissoc((ArrayNode *)self->root, 0, h, key, NULL, self->transient_id);
    } else {
        new_root = HashCollisionNode_dissoc((HashCollisionNode *)self->root, 0, h, key, NULL, self->transient_id);
    }

    if (!new_root) return NULL;

    if (new_root == self->root) {
        Py_DECREF(new_root);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_ssize_t new_cnt = (new_root != Py_None) ? self->cnt - 1 : 0;
    if (new_root == Py_None) {
        Py_DECREF(new_root);
        new_root = NULL;
    }

    Set *result = Set_create(new_cnt, new_root, self->transient_id);
    Py_XDECREF(new_root);
    return (PyObject *)result;
}

// SetIterator - yields only keys from the HAMT
struct SetIterator {
    PyObject_HEAD
    PyObject *kv_iter;  // Underlying key-value iterator
};

static PyTypeObject SetIteratorType;

static void SetIterator_dealloc(SetIterator *self) {
    Py_XDECREF(self->kv_iter);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SetIterator_next(SetIterator *self) {
    if (!self->kv_iter) return NULL;

    PyObject *pair = PyIter_Next(self->kv_iter);
    if (!pair) return NULL;  // StopIteration or error

    PyObject *key = PyTuple_GET_ITEM(pair, 0);
    Py_INCREF(key);
    Py_DECREF(pair);
    return key;
}

static PyTypeObject SetIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.SetIterator",
    .tp_basicsize = sizeof(SetIterator),
    .tp_dealloc = (destructor)SetIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)SetIterator_next,
};

static PyObject *Set_iter(Set *self) {
    if (self->root == NULL) {
        return PyObject_GetIter(PyList_New(0));
    }

    PyObject *kv_iter;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        kv_iter = BitmapIndexedNode_iter_kv((BitmapIndexedNode *)self->root);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        kv_iter = ArrayNode_iter_kv((ArrayNode *)self->root);
    } else {
        kv_iter = HashCollisionNode_iter_kv((HashCollisionNode *)self->root);
    }

    if (!kv_iter) return NULL;

    SetIterator *it = PyObject_New(SetIterator, &SetIteratorType);
    if (!it) {
        Py_DECREF(kv_iter);
        return NULL;
    }

    it->kv_iter = kv_iter;
    return (PyObject *)it;
}

// Set operations: union, intersection, difference

// Forward declarations for TransientSet functions (defined later)
static PyTypeObject TransientSetType;
static PyObject *Set_transient(Set *self, PyObject *Py_UNUSED(ignored));
static PyObject *TransientSet_conj_mut(TransientSet *self, PyObject *key);
static PyObject *TransientSet_disj_mut(TransientSet *self, PyObject *key);
static PyObject *TransientSet_persistent(TransientSet *self, PyObject *Py_UNUSED(ignored));

static PyObject *Set_or(PyObject *left, PyObject *right) {
    if (!PyObject_TypeCheck(left, &SetType)) {
        // Reflected union is supported for built-in set and frozenset values.
        if (!PyObject_TypeCheck(right, &SetType) || !PyAnySet_Check(left)) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        return Set_or(right, left);
    }

    Set *self = (Set *)left;
    if (!PyObject_TypeCheck(right, &SetType) && !PyAnySet_Check(right)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    // If other is also a Set, iterate directly
    if (PyObject_TypeCheck(right, &SetType)) {
        Set *other = (Set *)right;
        if (other->cnt == 0) {
            Py_INCREF(self);
            return (PyObject *)self;
        }
        if (self->cnt == 0) {
            Py_INCREF(other);
            return (PyObject *)other;
        }

        // Use the larger set as base and add elements from smaller
        Set *base = (self->cnt >= other->cnt) ? self : other;
        Set *to_add = (self->cnt >= other->cnt) ? other : self;

        // Create transient from base for efficient mutation
        TransientSet *trans = (TransientSet *)Set_transient(base, NULL);
        if (!trans) return NULL;

        PyObject *iter = Set_iter(to_add);
        if (!iter) {
            Py_DECREF(trans);
            return NULL;
        }

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            PyObject *res = TransientSet_conj_mut(trans, key);
            Py_DECREF(key);
            if (!res) {
                Py_DECREF(trans);
                Py_DECREF(iter);
                return NULL;
            }
            Py_DECREF(res);
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) {
            Py_DECREF(trans);
            return NULL;
        }

        PyObject *result = TransientSet_persistent(trans, NULL);
        Py_DECREF(trans);
        return result;
    }

    // Other is an iterable - use transient for efficient accumulation
    PyObject *iter = PyObject_GetIter(right);
    if (!iter) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        return NULL;
    }

    TransientSet *trans = (TransientSet *)Set_transient(self, NULL);
    if (!trans) {
        Py_DECREF(iter);
        return NULL;
    }

    PyObject *key;
    while ((key = PyIter_Next(iter)) != NULL) {
        PyObject *res = TransientSet_conj_mut(trans, key);
        Py_DECREF(key);
        if (!res) {
            Py_DECREF(trans);
            Py_DECREF(iter);
            return NULL;
        }
        Py_DECREF(res);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        Py_DECREF(trans);
        return NULL;
    }

    PyObject *result = TransientSet_persistent(trans, NULL);
    Py_DECREF(trans);
    return result;
}

static PyObject *Set_and(PyObject *left, PyObject *right) {
    if (!PyObject_TypeCheck(left, &SetType)) {
        // Intersection is commutative, so reflected operations can use the
        // persistent set as the implementation receiver.
        if (!PyObject_TypeCheck(right, &SetType) || !PyAnySet_Check(left)) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        return Set_and(right, left);
    }

    Set *self = (Set *)left;
    if (!PyObject_TypeCheck(right, &SetType) && !PyAnySet_Check(right)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self->cnt == 0) {
        Py_INCREF(EMPTY_SET);
        return (PyObject *)EMPTY_SET;
    }

    // Use transient for efficient accumulation from empty set
    TransientSet *trans = (TransientSet *)Set_transient(EMPTY_SET, NULL);
    if (!trans) return NULL;

    if (PyObject_TypeCheck(right, &SetType)) {
        Set *other = (Set *)right;
        if (other->cnt == 0) {
            Py_DECREF(trans);
            Py_INCREF(EMPTY_SET);
            return (PyObject *)EMPTY_SET;
        }

        // Iterate over smaller set, check membership in larger
        Set *smaller = (self->cnt <= other->cnt) ? self : other;
        Set *larger = (self->cnt <= other->cnt) ? other : self;

        PyObject *iter = Set_iter(smaller);
        if (!iter) {
            Py_DECREF(trans);
            return NULL;
        }

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(larger, key);
            if (found < 0) {
                Py_DECREF(key);
                Py_DECREF(iter);
                Py_DECREF(trans);
                return NULL;
            }
            if (found) {
                PyObject *res = TransientSet_conj_mut(trans, key);
                Py_DECREF(key);
                if (!res) {
                    Py_DECREF(iter);
                    Py_DECREF(trans);
                    return NULL;
                }
                Py_DECREF(res);
            } else {
                Py_DECREF(key);
            }
        }
        Py_DECREF(iter);
    } else {
        // Other is an iterable - check each element against self
        PyObject *iter = PyObject_GetIter(right);
        if (!iter) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                Py_DECREF(trans);
                Py_RETURN_NOTIMPLEMENTED;
            }
            Py_DECREF(trans);
            return NULL;
        }

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(self, key);
            if (found < 0) {
                Py_DECREF(key);
                Py_DECREF(iter);
                Py_DECREF(trans);
                return NULL;
            }
            if (found) {
                PyObject *res = TransientSet_conj_mut(trans, key);
                Py_DECREF(key);
                if (!res) {
                    Py_DECREF(iter);
                    Py_DECREF(trans);
                    return NULL;
                }
                Py_DECREF(res);
            } else {
                Py_DECREF(key);
            }
        }
        Py_DECREF(iter);
    }

    if (PyErr_Occurred()) {
        Py_DECREF(trans);
        return NULL;
    }

    PyObject *result = TransientSet_persistent(trans, NULL);
    Py_DECREF(trans);
    return result;
}

static PyObject *Set_sub(PyObject *left, PyObject *right) {
    if (!PyObject_TypeCheck(left, &SetType)) {
        // Difference is not commutative. Preserve operand order by first
        // converting a reflected built-in set to a persistent Set.
        if (!PyObject_TypeCheck(right, &SetType) || !PyAnySet_Check(left)) {
            Py_RETURN_NOTIMPLEMENTED;
        }

        PyObject *base = Set_or((PyObject *)EMPTY_SET, left);
        if (!base) return NULL;
        PyObject *result = Set_sub(base, right);
        Py_DECREF(base);
        return result;
    }

    Set *self = (Set *)left;
    if (!PyObject_TypeCheck(right, &SetType) && !PyAnySet_Check(right)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self->cnt == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    PyObject *iter;
    if (PyObject_TypeCheck(right, &SetType)) {
        iter = Set_iter((Set *)right);
    } else {
        iter = PyObject_GetIter(right);
    }

    if (!iter) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        return NULL;
    }

    // Use transient for efficient mutation
    TransientSet *trans = (TransientSet *)Set_transient(self, NULL);
    if (!trans) {
        Py_DECREF(iter);
        return NULL;
    }

    PyObject *key;
    while ((key = PyIter_Next(iter)) != NULL) {
        PyObject *res = TransientSet_disj_mut(trans, key);
        Py_DECREF(key);
        if (!res) {
            Py_DECREF(trans);
            Py_DECREF(iter);
            return NULL;
        }
        Py_DECREF(res);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        Py_DECREF(trans);
        return NULL;
    }

    PyObject *result = TransientSet_persistent(trans, NULL);
    Py_DECREF(trans);
    return result;
}

static PyObject *Set_xor(PyObject *left, PyObject *right) {
    // Symmetric difference: (self | other) - (self & other)
    // or equivalently: (self - other) | (other - self)
    if (!PyObject_TypeCheck(left, &SetType)) {
        if (!PyObject_TypeCheck(right, &SetType) || !PyAnySet_Check(left)) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        return Set_xor(right, left);
    }

    Set *self = (Set *)left;
    if (!PyObject_TypeCheck(right, &SetType) && !PyAnySet_Check(right)) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    // Use transient for efficient accumulation from empty set
    TransientSet *trans = (TransientSet *)Set_transient(EMPTY_SET, NULL);
    if (!trans) return NULL;

    // Add elements from self that are not in other
    PyObject *self_iter = Set_iter(self);
    if (!self_iter) {
        Py_DECREF(trans);
        return NULL;
    }

    PyObject *key;
    while ((key = PyIter_Next(self_iter)) != NULL) {
        int in_other;
        if (PyObject_TypeCheck(right, &SetType)) {
            in_other = Set_contains((Set *)right, key);
        } else {
            // For non-Set iterables, we need to check membership
            in_other = PySequence_Contains(right, key);
        }

        if (in_other < 0) {
            Py_DECREF(key);
            Py_DECREF(self_iter);
            Py_DECREF(trans);
            return NULL;
        }

        if (!in_other) {
            PyObject *res = TransientSet_conj_mut(trans, key);
            Py_DECREF(key);
            if (!res) {
                Py_DECREF(self_iter);
                Py_DECREF(trans);
                return NULL;
            }
            Py_DECREF(res);
        } else {
            Py_DECREF(key);
        }
    }
    Py_DECREF(self_iter);

    if (PyErr_Occurred()) {
        Py_DECREF(trans);
        return NULL;
    }

    // Add elements from other that are not in self
    PyObject *other_iter;
    if (PyObject_TypeCheck(right, &SetType)) {
        other_iter = Set_iter((Set *)right);
    } else {
        other_iter = PyObject_GetIter(right);
    }

    if (!other_iter) {
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_DECREF(trans);
            Py_RETURN_NOTIMPLEMENTED;
        }
        Py_DECREF(trans);
        return NULL;
    }

    while ((key = PyIter_Next(other_iter)) != NULL) {
        int in_self = Set_contains(self, key);
        if (in_self < 0) {
            Py_DECREF(key);
            Py_DECREF(other_iter);
            Py_DECREF(trans);
            return NULL;
        }

        if (!in_self) {
            PyObject *res = TransientSet_conj_mut(trans, key);
            Py_DECREF(key);
            if (!res) {
                Py_DECREF(other_iter);
                Py_DECREF(trans);
                return NULL;
            }
            Py_DECREF(res);
        } else {
            Py_DECREF(key);
        }
    }
    Py_DECREF(other_iter);

    if (PyErr_Occurred()) {
        Py_DECREF(trans);
        return NULL;
    }

    PyObject *result = TransientSet_persistent(trans, NULL);
    Py_DECREF(trans);
    return result;
}

static Py_hash_t Set_hash(Set *self) {
    if (self->hash_computed) {
        return self->hash;
    }

    // Use XOR of element hashes for order-independent hash
    Py_hash_t h = 0;
    PyObject *iter = Set_iter(self);
    if (!iter) return -1;

    PyObject *key;
    while ((key = PyIter_Next(iter)) != NULL) {
        Py_hash_t kh = PyObject_Hash(key);
        Py_DECREF(key);

        if (kh == -1 && PyErr_Occurred()) {
            Py_DECREF(iter);
            return -1;
        }

        h ^= kh;
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) return -1;

    // Avoid returning -1 which signals error
    if (h == -1) h = -2;

    self->hash = h;
    self->hash_computed = 1;
    return h;
}

static PyObject *Set_richcompare(Set *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE && op != Py_LT && op != Py_LE && op != Py_GT && op != Py_GE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (self == (Set *)other) {
        if (op == Py_EQ || op == Py_LE || op == Py_GE) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
    }

    if (!PyObject_TypeCheck(other, &SetType)) {
        if (op == Py_EQ) Py_RETURN_FALSE;
        if (op == Py_NE) Py_RETURN_TRUE;
        Py_RETURN_NOTIMPLEMENTED;
    }

    Set *o = (Set *)other;

    // For equality, check counts first
    if (op == Py_EQ || op == Py_NE) {
        if (self->cnt != o->cnt) {
            return PyBool_FromLong(op == Py_NE);
        }

        // Check all elements of self are in other
        PyObject *iter = Set_iter(self);
        if (!iter) return NULL;

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(o, key);
            Py_DECREF(key);

            if (found < 0) {
                Py_DECREF(iter);
                return NULL;
            }
            if (!found) {
                Py_DECREF(iter);
                return PyBool_FromLong(op == Py_NE);
            }
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) return NULL;

        return PyBool_FromLong(op == Py_EQ);
    }

    // Subset/superset comparisons
    // self < other: self is proper subset of other
    // self <= other: self is subset of other
    // self > other: self is proper superset of other
    // self >= other: self is superset of other

    if (op == Py_LT) {
        // self < other: cnt < other.cnt and all self elements in other
        if (self->cnt >= o->cnt) Py_RETURN_FALSE;

        PyObject *iter = Set_iter(self);
        if (!iter) return NULL;

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(o, key);
            Py_DECREF(key);

            if (found < 0) {
                Py_DECREF(iter);
                return NULL;
            }
            if (!found) {
                Py_DECREF(iter);
                Py_RETURN_FALSE;
            }
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) return NULL;
        Py_RETURN_TRUE;
    }

    if (op == Py_LE) {
        // self <= other: all self elements in other
        if (self->cnt > o->cnt) Py_RETURN_FALSE;

        PyObject *iter = Set_iter(self);
        if (!iter) return NULL;

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(o, key);
            Py_DECREF(key);

            if (found < 0) {
                Py_DECREF(iter);
                return NULL;
            }
            if (!found) {
                Py_DECREF(iter);
                Py_RETURN_FALSE;
            }
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) return NULL;
        Py_RETURN_TRUE;
    }

    if (op == Py_GT) {
        // self > other: other < self
        if (o->cnt >= self->cnt) Py_RETURN_FALSE;

        PyObject *iter = Set_iter(o);
        if (!iter) return NULL;

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(self, key);
            Py_DECREF(key);

            if (found < 0) {
                Py_DECREF(iter);
                return NULL;
            }
            if (!found) {
                Py_DECREF(iter);
                Py_RETURN_FALSE;
            }
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) return NULL;
        Py_RETURN_TRUE;
    }

    if (op == Py_GE) {
        // self >= other: other <= self
        if (o->cnt > self->cnt) Py_RETURN_FALSE;

        PyObject *iter = Set_iter(o);
        if (!iter) return NULL;

        PyObject *key;
        while ((key = PyIter_Next(iter)) != NULL) {
            int found = Set_contains(self, key);
            Py_DECREF(key);

            if (found < 0) {
                Py_DECREF(iter);
                return NULL;
            }
            if (!found) {
                Py_DECREF(iter);
                Py_RETURN_FALSE;
            }
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) return NULL;
        Py_RETURN_TRUE;
    }

    Py_RETURN_NOTIMPLEMENTED;
}

static PyObject *Set_repr(Set *self) {
    if (self->cnt == 0) {
        return PyUnicode_FromString("#{}");
    }

    PyObject *parts = PyList_New(0);
    if (!parts) return NULL;

    PyObject *iter = Set_iter(self);
    if (!iter) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *key;
    while ((key = PyIter_Next(iter)) != NULL) {
        PyObject *key_repr = PyObject_Repr(key);
        Py_DECREF(key);

        if (!key_repr) {
            Py_DECREF(iter);
            Py_DECREF(parts);
            return NULL;
        }

        if (PyList_Append(parts, key_repr) < 0) {
            Py_DECREF(key_repr);
            Py_DECREF(iter);
            Py_DECREF(parts);
            return NULL;
        }
        Py_DECREF(key_repr);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *space = PyUnicode_FromString(" ");
    if (!space) {
        Py_DECREF(parts);
        return NULL;
    }

    PyObject *joined = PyUnicode_Join(space, parts);
    Py_DECREF(space);
    Py_DECREF(parts);
    if (!joined) return NULL;

    PyObject *result = PyUnicode_FromFormat("#{%U}", joined);
    Py_DECREF(joined);
    return result;
}

// TransientSet forward declaration
static PyTypeObject TransientSetType;

static PyObject *Set_transient(Set *self, PyObject *Py_UNUSED(ignored));

static PyObject *Set_to_seq(Set *self, PyObject *Py_UNUSED(ignored)) {
    if (self->cnt == 0) {
        Py_RETURN_NONE;
    }

    // Build Cons list of elements
    PyObject *iter = Set_iter(self);
    if (!iter) return NULL;

    PyObject *elements = PyList_New(0);
    if (!elements) {
        Py_DECREF(iter);
        return NULL;
    }

    PyObject *key;
    while ((key = PyIter_Next(iter)) != NULL) {
        if (PyList_Append(elements, key) < 0) {
            Py_DECREF(key);
            Py_DECREF(iter);
            Py_DECREF(elements);
            return NULL;
        }
        Py_DECREF(key);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        Py_DECREF(elements);
        return NULL;
    }

    // Build Cons list in reverse
    Cons *result = NULL;
    for (Py_ssize_t i = PyList_Size(elements) - 1; i >= 0; i--) {
        PyObject *item = PyList_GET_ITEM(elements, i);
        Cons *new_cons = (Cons *)ConsType.tp_alloc(&ConsType, 0);
        if (!new_cons) {
            Py_XDECREF(result);
            Py_DECREF(elements);
            return NULL;
        }

        Py_INCREF(item);
        new_cons->first = item;
        new_cons->rest = result ? (PyObject *)result : Py_None;
        Py_INCREF(new_cons->rest);
        new_cons->hash = 0;
        new_cons->hash_computed = 0;

        result = new_cons;
    }

    Py_DECREF(elements);
    return (PyObject *)result;
}

/* Set.copy() - returns self since Set is immutable */
static PyObject *Set_copy(Set *self, PyObject *Py_UNUSED(ignored)) {
    Py_INCREF(self);
    return (PyObject *)self;
}

/* Set.isdisjoint(other) - return True if no common elements */
static PyObject *Set_isdisjoint(Set *self, PyObject *other) {
    PyObject *iter = PyObject_GetIter(other);
    if (!iter) return NULL;

    PyObject *item;
    while ((item = PyIter_Next(iter)) != NULL) {
        int contains = Set_contains(self, item);
        Py_DECREF(item);

        if (contains < 0) {
            Py_DECREF(iter);
            return NULL;
        }
        if (contains) {
            Py_DECREF(iter);
            Py_RETURN_FALSE;
        }
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) return NULL;

    Py_RETURN_TRUE;
}

static PyObject *Set_reduce(Set *self, PyObject *Py_UNUSED(ignored)) {
    // Set accepts one iterable constructor argument, so wrap the elements
    // tuple in the argument tuple used by pickle.
    PyObject *elements = PySequence_Tuple((PyObject *)self);
    if (!elements) return NULL;

    PyObject *args = PyTuple_Pack(1, elements);
    Py_DECREF(elements);
    if (!args) return NULL;

    PyObject *result = PyTuple_Pack(2, (PyObject *)Py_TYPE(self), args);
    Py_DECREF(args);
    return result;
}

static PyMethodDef Set_methods[] = {
    {"conj", (PyCFunction)Set_conj, METH_O, "Add element to set"},
    {"disj", (PyCFunction)Set_disj, METH_O, "Remove element from set"},
    {"transient", (PyCFunction)Set_transient, METH_NOARGS, "Get transient version"},
    {"to_seq", (PyCFunction)Set_to_seq, METH_NOARGS, "Convert to Cons sequence"},
    {"copy", (PyCFunction)Set_copy, METH_NOARGS, "Return self (immutable sets don't need copying)"},
    {"isdisjoint", (PyCFunction)Set_isdisjoint, METH_O, "Return True if no common elements with other"},
    {"__reduce__", (PyCFunction)Set_reduce, METH_NOARGS, "Pickle support"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations (e.g., Set[int])"},
    {NULL}
};

static PySequenceMethods Set_as_sequence = {
    .sq_length = (lenfunc)Set_length,
    .sq_contains = (objobjproc)Set_contains,
};

static PyNumberMethods Set_as_number = {
    .nb_or = Set_or,
    .nb_and = Set_and,
    .nb_subtract = Set_sub,
    .nb_xor = Set_xor,
};

static PyTypeObject SetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.Set",
    .tp_doc = "Set(iterable=()) -> persistent hash set using a HAMT",
    .tp_basicsize = sizeof(Set),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Set_dealloc,
    .tp_repr = (reprfunc)Set_repr,
    .tp_as_number = &Set_as_number,
    .tp_as_sequence = &Set_as_sequence,
    .tp_hash = (hashfunc)Set_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_richcompare = (richcmpfunc)Set_richcompare,
    .tp_iter = (getiterfunc)Set_iter,
    .tp_methods = Set_methods,
    .tp_init = (initproc)Set_init,
    .tp_new = Set_new,
};

// === TransientSet ===
struct TransientSet {
    PyObject_HEAD
    Py_ssize_t cnt;
    PyObject *root;
    PyObject *id;
};

static void TransientSet_dealloc(TransientSet *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Set_transient(Set *self, PyObject *Py_UNUSED(ignored)) {
    TransientSet *t = PyObject_New(TransientSet, &TransientSetType);
    if (!t) return NULL;

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        return NULL;
    }

    t->cnt = self->cnt;
    if (self->root != NULL) {
        if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
            t->root = (PyObject *)BitmapIndexedNode_ensure_editable((BitmapIndexedNode *)self->root, t->id);
        } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
            t->root = (PyObject *)ArrayNode_ensure_editable((ArrayNode *)self->root, t->id);
        } else {
            t->root = (PyObject *)HashCollisionNode_ensure_editable((HashCollisionNode *)self->root, t->id);
        }
        if (!t->root) {
            Py_DECREF(t);
            return NULL;
        }
    } else {
        t->root = NULL;
    }

    return (PyObject *)t;
}

static void TransientSet_ensure_editable(TransientSet *self) {
    if (self->id == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Transient used after persistent() call");
    }
}

static PyObject *TransientSet_conj_mut(TransientSet *self, PyObject *key) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    PyObject *added_leaf = PyList_New(0);
    if (!added_leaf) return NULL;

    PyObject *root = self->root ? self->root : (PyObject *)EMPTY_BIN;
    PyObject *new_root;

    if (PyObject_TypeCheck(root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_assoc((BitmapIndexedNode *)root, 0, h, key, Py_None, added_leaf, self->id);
    } else if (PyObject_TypeCheck(root, &ArrayNodeType)) {
        new_root = ArrayNode_assoc((ArrayNode *)root, 0, h, key, Py_None, added_leaf, self->id);
    } else {
        new_root = HashCollisionNode_assoc((HashCollisionNode *)root, 0, h, key, Py_None, added_leaf, self->id);
    }

    if (!new_root) {
        Py_DECREF(added_leaf);
        return NULL;
    }

    if (new_root != self->root) {
        Py_XDECREF(self->root);
        self->root = new_root;
    } else {
        Py_DECREF(new_root);
    }

    self->cnt += PyList_Size(added_leaf);
    Py_DECREF(added_leaf);

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientSet_disj_mut(TransientSet *self, PyObject *key) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return NULL;

    // Check if key exists before attempting removal
    // This is necessary because transient mutation may return the same pointer
    // even when a modification was made (due to in-place editing)
    PyObject *found;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        found = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        found = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        found = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!found) {
        // Error occurred during find
        return NULL;
    }
    if (found == _MISSING) {
        // Key not in set, nothing to do
        Py_DECREF(found);
        Py_INCREF(self);
        return (PyObject *)self;
    }
    Py_DECREF(found);

    // Key exists, perform removal
    PyObject *new_root;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        new_root = BitmapIndexedNode_dissoc((BitmapIndexedNode *)self->root, 0, h, key, NULL, self->id);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        new_root = ArrayNode_dissoc((ArrayNode *)self->root, 0, h, key, NULL, self->id);
    } else {
        new_root = HashCollisionNode_dissoc((HashCollisionNode *)self->root, 0, h, key, NULL, self->id);
    }

    if (!new_root && PyErr_Occurred()) {
        return NULL;
    }

    if (new_root == Py_None) {
        Py_DECREF(new_root);
        Py_XDECREF(self->root);
        self->root = NULL;
        self->cnt = 0;
    } else if (new_root != self->root) {
        Py_XDECREF(self->root);
        self->root = new_root;
        self->cnt--;
    } else {
        // Same pointer returned due to in-place mutation, but key was removed
        Py_DECREF(new_root);
        self->cnt--;
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientSet_persistent(TransientSet *self, PyObject *Py_UNUSED(ignored)) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_CLEAR(self->id);

    Set *result = Set_create(self->cnt, self->root, NULL);
    return (PyObject *)result;
}

// === TransientSet MutableSet Protocol ===

static Py_ssize_t TransientSet_length(TransientSet *self) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return -1;
    return self->cnt;
}

static int TransientSet_contains(TransientSet *self, PyObject *key) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return -1;

    if (self->root == NULL) {
        return 0;
    }

    Py_hash_t h = PyObject_Hash(key);
    if (h == -1 && PyErr_Occurred()) return -1;

    PyObject *result;
    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        result = BitmapIndexedNode_find((BitmapIndexedNode *)self->root, 0, h, key, _MISSING);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        result = ArrayNode_find((ArrayNode *)self->root, 0, h, key, _MISSING);
    } else {
        result = HashCollisionNode_find((HashCollisionNode *)self->root, 0, h, key, _MISSING);
    }

    if (!result) return -1;

    int found = (result != _MISSING);
    Py_DECREF(result);
    return found;
}

static PyObject *TransientSet_iter(TransientSet *self) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        // Return empty iterator
        return PyObject_GetIter(PyTuple_New(0));
    }

    if (PyObject_TypeCheck(self->root, &BitmapIndexedNodeType)) {
        return BitmapIndexedNode_iter_mode((BitmapIndexedNode *)self->root, ITER_MODE_KEYS);
    } else if (PyObject_TypeCheck(self->root, &ArrayNodeType)) {
        return ArrayNode_iter_mode((ArrayNode *)self->root, ITER_MODE_KEYS);
    } else {
        return HashCollisionNode_iter_mode((HashCollisionNode *)self->root, ITER_MODE_KEYS);
    }
}

// Python set methods: add, discard, remove, clear

static PyObject *TransientSet_add(TransientSet *self, PyObject *key) {
    // add() is alias for conj_mut but returns None (Python set semantics)
    PyObject *result = TransientSet_conj_mut(self, key);
    if (!result) return NULL;
    Py_DECREF(result);
    Py_RETURN_NONE;
}

static PyObject *TransientSet_discard(TransientSet *self, PyObject *key) {
    // discard() removes key if present, does not raise if missing
    PyObject *result = TransientSet_disj_mut(self, key);
    if (!result) return NULL;
    Py_DECREF(result);
    Py_RETURN_NONE;
}

static PyObject *TransientSet_remove(TransientSet *self, PyObject *key) {
    // remove() raises KeyError if key not present
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    // Check if key exists first
    int found = TransientSet_contains(self, key);
    if (found < 0) return NULL;  // Error occurred

    if (!found) {
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }

    PyObject *result = TransientSet_disj_mut(self, key);
    if (!result) return NULL;
    Py_DECREF(result);
    Py_RETURN_NONE;
}

static PyObject *TransientSet_clear(TransientSet *self, PyObject *Py_UNUSED(ignored)) {
    TransientSet_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    Py_XDECREF(self->root);
    self->root = NULL;
    self->cnt = 0;

    Py_RETURN_NONE;
}

static PyMethodDef TransientSet_methods[] = {
    {"conj_mut", (PyCFunction)TransientSet_conj_mut, METH_O, "Mutably add element"},
    {"disj_mut", (PyCFunction)TransientSet_disj_mut, METH_O, "Mutably remove element"},
    {"persistent", (PyCFunction)TransientSet_persistent, METH_NOARGS, "Return persistent set"},
    {"add", (PyCFunction)TransientSet_add, METH_O, "Add element to set"},
    {"discard", (PyCFunction)TransientSet_discard, METH_O, "Remove element if present"},
    {"remove", (PyCFunction)TransientSet_remove, METH_O, "Remove element, raise KeyError if missing"},
    {"clear", (PyCFunction)TransientSet_clear, METH_NOARGS, "Remove all elements"},
    {NULL}
};

static PySequenceMethods TransientSet_as_sequence = {
    .sq_length = (lenfunc)TransientSet_length,
    .sq_contains = (objobjproc)TransientSet_contains,
};

static PyTypeObject TransientSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientSet",
    .tp_doc = "Transient set for batch operations",
    .tp_basicsize = sizeof(TransientSet),
    .tp_dealloc = (destructor)TransientSet_dealloc,
    .tp_as_sequence = &TransientSet_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = (getiterfunc)TransientSet_iter,
    .tp_methods = TransientSet_methods,
};

// === SortedVector (Persistent Red-Black Tree with size annotations) ===

#define RB_RED 0
#define RB_BLACK 1

// Red-Black Tree Node
typedef struct RBNode {
    PyObject_HEAD
    PyObject *value;       // The stored element
    PyObject *sort_key;    // Cached key for comparison (result of key_fn(value))
    struct RBNode *left;
    struct RBNode *right;
    Py_ssize_t size;       // Subtree size for O(log n) indexing
    unsigned char color;   // RB_RED or RB_BLACK
    PyObject *edit;        // For transient support (NULL = persistent)
} RBNode;

static PyTypeObject RBNodeType;

static void RBNode_dealloc(RBNode *self) {
    Py_XDECREF(self->value);
    Py_XDECREF(self->sort_key);
    Py_XDECREF(self->left);
    Py_XDECREF(self->right);
    Py_XDECREF(self->edit);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static Py_ssize_t RBNode_size(RBNode *node) {
    return node ? node->size : 0;
}

static void RBNode_update_size(RBNode *node) {
    if (node) {
        node->size = 1 + RBNode_size(node->left) + RBNode_size(node->right);
    }
}

static RBNode *RBNode_create(PyObject *value, PyObject *sort_key, unsigned char color, PyObject *edit) {
    RBNode *node = PyObject_New(RBNode, &RBNodeType);
    if (!node) return NULL;

    Py_INCREF(value);
    node->value = value;
    Py_INCREF(sort_key);
    node->sort_key = sort_key;
    node->left = NULL;
    node->right = NULL;
    node->size = 1;
    node->color = color;
    node->edit = edit;
    Py_XINCREF(edit);

    return node;
}

// Clone a node (for persistent operations)
static RBNode *RBNode_clone(RBNode *node, PyObject *edit) {
    if (!node) return NULL;

    RBNode *new_node = PyObject_New(RBNode, &RBNodeType);
    if (!new_node) return NULL;

    Py_INCREF(node->value);
    new_node->value = node->value;
    Py_INCREF(node->sort_key);
    new_node->sort_key = node->sort_key;
    new_node->left = node->left;
    Py_XINCREF(new_node->left);
    new_node->right = node->right;
    Py_XINCREF(new_node->right);
    new_node->size = node->size;
    new_node->color = node->color;
    new_node->edit = edit;
    Py_XINCREF(edit);

    return new_node;
}

static int RBNode_is_red(RBNode *node) {
    return node && node->color == RB_RED;
}

static int RBNode_is_editable(RBNode *node, PyObject *edit) {
    return node && edit && node->edit == edit;
}

// Ensure node is editable (clone if necessary)
static RBNode *RBNode_ensure_editable(RBNode *node, PyObject *edit) {
    if (!node) return NULL;
    if (RBNode_is_editable(node, edit)) {
        Py_INCREF(node);
        return node;
    }
    return RBNode_clone(node, edit);
}

static PyTypeObject RBNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.RBNode",
    .tp_doc = "Red-Black Tree Node (internal)",
    .tp_basicsize = sizeof(RBNode),
    .tp_dealloc = (destructor)RBNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

// Compare two sort keys, returning -1, 0, or 1
// Also handles reverse ordering
static int SortedVector_compare_keys(PyObject *a, PyObject *b, int reverse) {
    int cmp = PyObject_RichCompareBool(a, b, Py_LT);
    if (cmp < 0) return -2;  // Error
    if (cmp) return reverse ? 1 : -1;

    cmp = PyObject_RichCompareBool(a, b, Py_GT);
    if (cmp < 0) return -2;  // Error
    if (cmp) return reverse ? -1 : 1;

    return 0;  // Equal
}

// === Red-Black Tree Rotations (return new root of subtree) ===

static RBNode *RBNode_rotate_left(RBNode *h, PyObject *edit) {
    RBNode *x = RBNode_ensure_editable(h->right, edit);
    if (!x) return NULL;

    RBNode *new_h = RBNode_ensure_editable(h, edit);
    if (!new_h) {
        Py_DECREF(x);
        return NULL;
    }

    Py_XDECREF(new_h->right);
    new_h->right = x->left;
    Py_XINCREF(new_h->right);

    Py_XDECREF(x->left);
    x->left = new_h;
    // new_h reference is now owned by x->left

    x->color = new_h->color;
    new_h->color = RB_RED;

    RBNode_update_size(new_h);
    RBNode_update_size(x);

    return x;
}

static RBNode *RBNode_rotate_right(RBNode *h, PyObject *edit) {
    RBNode *x = RBNode_ensure_editable(h->left, edit);
    if (!x) return NULL;

    RBNode *new_h = RBNode_ensure_editable(h, edit);
    if (!new_h) {
        Py_DECREF(x);
        return NULL;
    }

    Py_XDECREF(new_h->left);
    new_h->left = x->right;
    Py_XINCREF(new_h->left);

    Py_XDECREF(x->right);
    x->right = new_h;
    // new_h reference is now owned by x->right

    x->color = new_h->color;
    new_h->color = RB_RED;

    RBNode_update_size(new_h);
    RBNode_update_size(x);

    return x;
}

// Safe flip colors that ensures all nodes are editable first
static RBNode *RBNode_flip_colors(RBNode *h, PyObject *edit) {
    RBNode *new_h = RBNode_ensure_editable(h, edit);
    if (!new_h) return NULL;

    new_h->color = !new_h->color;

    if (new_h->left) {
        RBNode *new_left = RBNode_ensure_editable(new_h->left, edit);
        if (!new_left) {
            Py_DECREF(new_h);
            return NULL;
        }
        new_left->color = !new_left->color;
        Py_XDECREF(new_h->left);
        new_h->left = new_left;
    }

    if (new_h->right) {
        RBNode *new_right = RBNode_ensure_editable(new_h->right, edit);
        if (!new_right) {
            Py_DECREF(new_h);
            return NULL;
        }
        new_right->color = !new_right->color;
        Py_XDECREF(new_h->right);
        new_h->right = new_right;
    }

    return new_h;
}

// Balance after insertion
static RBNode *RBNode_balance(RBNode *h, PyObject *edit) {
    if (!h) return NULL;

    // Right-leaning red link -> rotate left
    if (RBNode_is_red(h->right) && !RBNode_is_red(h->left)) {
        h = RBNode_rotate_left(h, edit);
        if (!h) return NULL;
    }

    // Two consecutive left red links -> rotate right
    if (RBNode_is_red(h->left) && RBNode_is_red(h->left->left)) {
        h = RBNode_rotate_right(h, edit);
        if (!h) return NULL;
    }

    // Both children red -> flip colors
    if (RBNode_is_red(h->left) && RBNode_is_red(h->right)) {
        h = RBNode_flip_colors(h, edit);
        if (!h) return NULL;
    }

    RBNode_update_size(h);
    return h;
}

// === SortedVector ===

typedef struct SortedVector {
    PyObject_HEAD
    RBNode *root;
    Py_ssize_t cnt;
    PyObject *key_fn;    // Optional key function (NULL = use element itself)
    int reverse;         // Sort in descending order
} SortedVector;

static PyTypeObject SortedVectorType;

static void SortedVector_dealloc(SortedVector *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->key_fn);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SortedVector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    SortedVector *self = (SortedVector *)type->tp_alloc(type, 0);
    if (self) {
        self->root = NULL;
        self->cnt = 0;
        self->key_fn = NULL;
        self->reverse = 0;
    }
    return (PyObject *)self;
}

// Get the sort key for a value
static PyObject *SortedVector_get_sort_key(SortedVector *self, PyObject *value) {
    if (self->key_fn && self->key_fn != Py_None) {
        return PyObject_CallOneArg(self->key_fn, value);
    }
    Py_INCREF(value);
    return value;
}

// Recursive insert into Red-Black Tree
static RBNode *RBNode_insert(RBNode *h, PyObject *value, PyObject *sort_key, int reverse, PyObject *edit) {
    if (!h) {
        return RBNode_create(value, sort_key, RB_RED, edit);
    }

    int cmp = SortedVector_compare_keys(sort_key, h->sort_key, reverse);
    if (cmp == -2) return NULL;  // Error

    RBNode *new_h = RBNode_ensure_editable(h, edit);
    if (!new_h) return NULL;

    if (cmp < 0) {
        RBNode *new_left = RBNode_insert(new_h->left, value, sort_key, reverse, edit);
        if (!new_left && PyErr_Occurred()) {
            Py_DECREF(new_h);
            return NULL;
        }
        Py_XDECREF(new_h->left);
        new_h->left = new_left;
    } else {
        // cmp >= 0: equal keys go to the right to maintain insertion order for duplicates
        RBNode *new_right = RBNode_insert(new_h->right, value, sort_key, reverse, edit);
        if (!new_right && PyErr_Occurred()) {
            Py_DECREF(new_h);
            return NULL;
        }
        Py_XDECREF(new_h->right);
        new_h->right = new_right;
    }

    return RBNode_balance(new_h, edit);
}

// conj: Add element maintaining sorted order
static PyObject *SortedVector_conj(SortedVector *self, PyObject *value) {
    PyObject *sort_key = SortedVector_get_sort_key(self, value);
    if (!sort_key) return NULL;

    RBNode *new_root = RBNode_insert(self->root, value, sort_key, self->reverse, NULL);
    Py_DECREF(sort_key);

    if (!new_root && PyErr_Occurred()) return NULL;

    // Make root black
    if (new_root && new_root->color == RB_RED) {
        RBNode *black_root = RBNode_clone(new_root, NULL);
        Py_DECREF(new_root);
        if (!black_root) return NULL;
        black_root->color = RB_BLACK;
        new_root = black_root;
    }

    // Create new SortedVector
    SortedVector *result = PyObject_New(SortedVector, &SortedVectorType);
    if (!result) {
        Py_XDECREF(new_root);
        return NULL;
    }

    result->root = new_root;
    result->cnt = self->cnt + 1;
    result->key_fn = self->key_fn;
    Py_XINCREF(result->key_fn);
    result->reverse = self->reverse;

    return (PyObject *)result;
}

// nth: Get element at index (O(log n))
static PyObject *RBNode_nth(RBNode *node, Py_ssize_t index) {
    if (!node) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    Py_ssize_t left_size = RBNode_size(node->left);

    if (index < left_size) {
        return RBNode_nth(node->left, index);
    } else if (index == left_size) {
        Py_INCREF(node->value);
        return node->value;
    } else {
        return RBNode_nth(node->right, index - left_size - 1);
    }
}

static PyObject *SortedVector_nth(SortedVector *self, PyObject *args) {
    Py_ssize_t index;
    PyObject *default_val = NULL;

    if (!PyArg_ParseTuple(args, "n|O", &index, &default_val)) {
        return NULL;
    }

    // Handle negative indices
    if (index < 0) {
        index += self->cnt;
    }

    if (index < 0 || index >= self->cnt) {
        if (default_val) {
            Py_INCREF(default_val);
            return default_val;
        }
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    return RBNode_nth(self->root, index);
}

static PyObject *SortedVector_getitem(SortedVector *self, PyObject *key) {
    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        if (index == -1 && PyErr_Occurred()) return NULL;

        if (index < 0) index += self->cnt;

        if (index < 0 || index >= self->cnt) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }

        return RBNode_nth(self->root, index);
    }

    PyErr_SetString(PyExc_TypeError, "indices must be integers");
    return NULL;
}

static Py_ssize_t SortedVector_length(SortedVector *self) {
    return self->cnt;
}

// first: Get minimum element
static PyObject *SortedVector_first(SortedVector *self, PyObject *Py_UNUSED(ignored)) {
    if (self->cnt == 0) {
        Py_RETURN_NONE;
    }

    RBNode *node = self->root;
    while (node->left) {
        node = node->left;
    }

    Py_INCREF(node->value);
    return node->value;
}

// last: Get maximum element
static PyObject *SortedVector_last(SortedVector *self, PyObject *Py_UNUSED(ignored)) {
    if (self->cnt == 0) {
        Py_RETURN_NONE;
    }

    RBNode *node = self->root;
    while (node->right) {
        node = node->right;
    }

    Py_INCREF(node->value);
    return node->value;
}

// Binary search for a value, returns index or -1 if not found
static Py_ssize_t RBNode_index_of(RBNode *node, PyObject *sort_key, PyObject *value,
                                      int reverse, Py_ssize_t offset) {
    if (!node) return -1;

    int cmp = SortedVector_compare_keys(sort_key, node->sort_key, reverse);
    if (cmp == -2) return -2;  // Error

    Py_ssize_t left_size = RBNode_size(node->left);

    if (cmp < 0) {
        return RBNode_index_of(node->left, sort_key, value, reverse, offset);
    } else if (cmp > 0) {
        return RBNode_index_of(node->right, sort_key, value, reverse,
                               offset + left_size + 1);
    } else {
        // Equal sort keys can belong to unequal values. Rotations may place
        // them on either side, so search the whole equal-key range and return
        // the first exact value match.
        Py_ssize_t left_result = RBNode_index_of(node->left, sort_key, value,
                                                  reverse, offset);
        if (left_result == -2) return -2;  // Error
        if (left_result >= 0) return left_result;

        int eq = PyObject_RichCompareBool(value, node->value, Py_EQ);
        if (eq < 0) return -2;
        if (eq) return offset + left_size;

        return RBNode_index_of(node->right, sort_key, value, reverse,
                               offset + left_size + 1);
    }
}

static PyObject *SortedVector_index_of(SortedVector *self, PyObject *value) {
    PyObject *sort_key = SortedVector_get_sort_key(self, value);
    if (!sort_key) return NULL;

    Py_ssize_t index = RBNode_index_of(self->root, sort_key, value,
                                       self->reverse, 0);
    Py_DECREF(sort_key);

    if (index == -2) return NULL;  // Error occurred

    return PyLong_FromSsize_t(index);
}

// contains: Check if element exists
static int RBNode_contains(RBNode *node, PyObject *sort_key, PyObject *value, int reverse) {
    if (!node) return 0;

    int cmp = SortedVector_compare_keys(sort_key, node->sort_key, reverse);
    if (cmp == -2) return -1;  // Error

    if (cmp < 0) {
        return RBNode_contains(node->left, sort_key, value, reverse);
    } else if (cmp > 0) {
        return RBNode_contains(node->right, sort_key, value, reverse);
    } else {
        // Keys match - check if values are equal
        int eq = PyObject_RichCompareBool(value, node->value, Py_EQ);
        if (eq < 0) return -1;
        if (eq) return 1;

        // Check both subtrees for equal keys
        int left_result = RBNode_contains(node->left, sort_key, value, reverse);
        if (left_result != 0) return left_result;
        return RBNode_contains(node->right, sort_key, value, reverse);
    }
}

static int SortedVector_contains(SortedVector *self, PyObject *value) {
    if (self->cnt == 0) return 0;

    PyObject *sort_key = SortedVector_get_sort_key(self, value);
    if (!sort_key) return -1;

    int result = RBNode_contains(self->root, sort_key, value, self->reverse);
    Py_DECREF(sort_key);

    return result;
}

// rank: Count of elements less than given value
static Py_ssize_t RBNode_rank(RBNode *node, PyObject *sort_key, int reverse) {
    if (!node) return 0;

    int cmp = SortedVector_compare_keys(sort_key, node->sort_key, reverse);
    if (cmp == -2) return -1;  // Error

    Py_ssize_t left_size = RBNode_size(node->left);

    if (cmp <= 0) {
        return RBNode_rank(node->left, sort_key, reverse);
    } else {
        Py_ssize_t right_rank = RBNode_rank(node->right, sort_key, reverse);
        if (right_rank < 0) return -1;
        return left_size + 1 + right_rank;
    }
}

static PyObject *SortedVector_rank(SortedVector *self, PyObject *value) {
    PyObject *sort_key = SortedVector_get_sort_key(self, value);
    if (!sort_key) return NULL;

    Py_ssize_t r = RBNode_rank(self->root, sort_key, self->reverse);
    Py_DECREF(sort_key);

    if (r < 0) return NULL;
    return PyLong_FromSsize_t(r);
}

// === SortedVector Iterator (in-order traversal) ===

typedef struct SortedVectorIterator {
    PyObject_HEAD
    RBNode **stack;
    Py_ssize_t stack_size;
    Py_ssize_t stack_capacity;
} SortedVectorIterator;

static PyTypeObject SortedVectorIteratorType;

static void SortedVectorIterator_dealloc(SortedVectorIterator *self) {
    for (Py_ssize_t i = 0; i < self->stack_size; i++) {
        Py_XDECREF(self->stack[i]);
    }
    PyMem_Free(self->stack);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void SortedVectorIterator_push_left(SortedVectorIterator *self, RBNode *node) {
    while (node) {
        if (self->stack_size >= self->stack_capacity) {
            Py_ssize_t new_cap = self->stack_capacity * 2;
            RBNode **new_stack = PyMem_Realloc(self->stack, new_cap * sizeof(RBNode *));
            if (!new_stack) return;
            self->stack = new_stack;
            self->stack_capacity = new_cap;
        }
        Py_INCREF(node);
        self->stack[self->stack_size++] = node;
        node = node->left;
    }
}

static PyObject *SortedVectorIterator_next(SortedVectorIterator *self) {
    if (self->stack_size == 0) {
        return NULL;  // StopIteration
    }

    RBNode *node = self->stack[--self->stack_size];
    PyObject *value = node->value;
    Py_INCREF(value);

    // Push left spine of right subtree
    SortedVectorIterator_push_left(self, node->right);

    Py_DECREF(node);
    return value;
}

static PyTypeObject SortedVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.SortedVectorIterator",
    .tp_basicsize = sizeof(SortedVectorIterator),
    .tp_dealloc = (destructor)SortedVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)SortedVectorIterator_next,
};

static PyObject *SortedVector_iter(SortedVector *self) {
    SortedVectorIterator *iter = PyObject_New(SortedVectorIterator, &SortedVectorIteratorType);
    if (!iter) return NULL;

    // Initial capacity based on expected tree depth
    iter->stack_capacity = 32;
    iter->stack = PyMem_Malloc(iter->stack_capacity * sizeof(RBNode *));
    if (!iter->stack) {
        Py_DECREF(iter);
        return PyErr_NoMemory();
    }
    iter->stack_size = 0;

    SortedVectorIterator_push_left(iter, self->root);

    return (PyObject *)iter;
}

// === SortedVector repr ===

static PyObject *SortedVector_repr(SortedVector *self) {
    if (self->cnt == 0) {
        return PyUnicode_FromString("sorted_vec()");
    }

    PyObject *items = PyList_New(0);
    if (!items) return NULL;

    PyObject *iter = SortedVector_iter(self);
    if (!iter) {
        Py_DECREF(items);
        return NULL;
    }

    PyObject *item;
    while ((item = SortedVectorIterator_next((SortedVectorIterator *)iter)) != NULL) {
        PyObject *repr = PyObject_Repr(item);
        Py_DECREF(item);
        if (!repr) {
            Py_DECREF(iter);
            Py_DECREF(items);
            return NULL;
        }
        if (PyList_Append(items, repr) < 0) {
            Py_DECREF(repr);
            Py_DECREF(iter);
            Py_DECREF(items);
            return NULL;
        }
        Py_DECREF(repr);
    }
    Py_DECREF(iter);

    PyObject *sep = PyUnicode_FromString(", ");
    if (!sep) {
        Py_DECREF(items);
        return NULL;
    }

    PyObject *joined = PyUnicode_Join(sep, items);
    Py_DECREF(sep);
    Py_DECREF(items);
    if (!joined) return NULL;

    PyObject *result = PyUnicode_FromFormat("sorted_vec(%U)", joined);
    Py_DECREF(joined);
    return result;
}

// === SortedVector hash ===

static Py_hash_t SortedVector_hash(SortedVector *self) {
    Py_hash_t hash = 0x345678;
    Py_hash_t mult = 1000003;

    PyObject *iter = SortedVector_iter(self);
    if (!iter) return -1;

    PyObject *item;
    while ((item = SortedVectorIterator_next((SortedVectorIterator *)iter)) != NULL) {
        Py_hash_t item_hash = PyObject_Hash(item);
        Py_DECREF(item);
        if (item_hash == -1) {
            Py_DECREF(iter);
            return -1;
        }
        hash = (hash ^ item_hash) * mult;
        mult += 82520 + 2 * self->cnt;
    }
    Py_DECREF(iter);

    hash += 97531;
    if (hash == -1) hash = -2;
    return hash;
}

// === SortedVector equality ===

static PyObject *SortedVector_richcompare(SortedVector *self, PyObject *other, int op) {
    if (op != Py_EQ && op != Py_NE) {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (!PyObject_TypeCheck(other, &SortedVectorType)) {
        if (op == Py_EQ) Py_RETURN_FALSE;
        Py_RETURN_TRUE;
    }

    SortedVector *other_sv = (SortedVector *)other;

    if (self->cnt != other_sv->cnt) {
        if (op == Py_EQ) Py_RETURN_FALSE;
        Py_RETURN_TRUE;
    }

    // Compare elements in order
    PyObject *iter1 = SortedVector_iter(self);
    PyObject *iter2 = SortedVector_iter(other_sv);
    if (!iter1 || !iter2) {
        Py_XDECREF(iter1);
        Py_XDECREF(iter2);
        return NULL;
    }

    int equal = 1;
    PyObject *item1, *item2;
    while ((item1 = SortedVectorIterator_next((SortedVectorIterator *)iter1)) != NULL) {
        item2 = SortedVectorIterator_next((SortedVectorIterator *)iter2);
        if (!item2) {
            Py_DECREF(item1);
            equal = 0;
            break;
        }

        int cmp = PyObject_RichCompareBool(item1, item2, Py_EQ);
        Py_DECREF(item1);
        Py_DECREF(item2);

        if (cmp < 0) {
            Py_DECREF(iter1);
            Py_DECREF(iter2);
            return NULL;
        }
        if (!cmp) {
            equal = 0;
            break;
        }
    }

    Py_DECREF(iter1);
    Py_DECREF(iter2);

    if (op == Py_EQ) {
        if (equal) Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    } else {
        if (equal) Py_RETURN_FALSE;
        Py_RETURN_TRUE;
    }
}

// === SortedVector disj (remove element) ===

// Helper functions for Red-Black deletion
static RBNode *RBNode_move_red_left(RBNode *h, PyObject *edit) {
    h = RBNode_flip_colors(h, edit);
    if (!h) return NULL;

    if (h->right && RBNode_is_red(h->right->left)) {
        RBNode *new_right = RBNode_rotate_right(h->right, edit);
        if (!new_right) {
            Py_DECREF(h);
            return NULL;
        }
        Py_XDECREF(h->right);
        h->right = new_right;

        h = RBNode_rotate_left(h, edit);
        if (!h) return NULL;
        h = RBNode_flip_colors(h, edit);
        if (!h) return NULL;
    }
    return h;
}

static RBNode *RBNode_move_red_right(RBNode *h, PyObject *edit) {
    h = RBNode_flip_colors(h, edit);
    if (!h) return NULL;

    if (h->left && RBNode_is_red(h->left->left)) {
        h = RBNode_rotate_right(h, edit);
        if (!h) return NULL;
        h = RBNode_flip_colors(h, edit);
        if (!h) return NULL;
    }
    return h;
}

static RBNode *RBNode_min(RBNode *node) {
    while (node->left) node = node->left;
    return node;
}

static RBNode *RBNode_delete_min(RBNode *h, PyObject *edit) {
    // First ensure we have an editable copy before any modifications
    RBNode *new_h = RBNode_ensure_editable(h, edit);
    if (!new_h) return NULL;

    // Now check if this is the min node (no left child). Equal-key
    // duplicates can give the minimum node a right subtree, which must be
    // retained when the node is removed.
    if (!new_h->left) {
        RBNode *replacement = new_h->right;
        Py_XINCREF(replacement);
        Py_DECREF(new_h);
        return replacement;
    }

    if (!RBNode_is_red(new_h->left) && !RBNode_is_red(new_h->left->left)) {
        new_h = RBNode_move_red_left(new_h, edit);
        if (!new_h) return NULL;
    }

    RBNode *new_left = RBNode_delete_min(new_h->left, edit);
    if (new_h->left && !new_left && PyErr_Occurred()) {
        Py_DECREF(new_h);
        return NULL;
    }
    Py_XDECREF(new_h->left);
    new_h->left = new_left;

    return RBNode_balance(new_h, edit);
}

// Delete a node with matching sort_key and value
static RBNode *RBNode_delete(RBNode *h, PyObject *sort_key, PyObject *value, int reverse, int *deleted, PyObject *edit) {
    if (!h) {
        *deleted = 0;
        return NULL;
    }

    int cmp = SortedVector_compare_keys(sort_key, h->sort_key, reverse);
    if (cmp == -2) return NULL;  // Error

    if (cmp == 0) {
        // Values with equal sort keys may appear on either side after tree
        // rotations. Choose the subtree containing the exact value rather
        // than assuming every equal-key value is interchangeable.
        int eq = PyObject_RichCompareBool(value, h->value, Py_EQ);
        if (eq < 0) return NULL;
        if (!eq) {
            int in_left = RBNode_contains(h->left, sort_key, value, reverse);
            if (in_left < 0) return NULL;
            cmp = in_left ? -1 : 1;
        }
    }

    RBNode *new_h = RBNode_ensure_editable(h, edit);
    if (!new_h) return NULL;

    if (cmp < 0) {
        if (new_h->left && !RBNode_is_red(new_h->left) && !RBNode_is_red(new_h->left->left)) {
            new_h = RBNode_move_red_left(new_h, edit);
            if (!new_h) return NULL;
        }
        RBNode *new_left = RBNode_delete(new_h->left, sort_key, value, reverse, deleted, edit);
        if (new_h->left && !new_left && PyErr_Occurred()) {
            Py_DECREF(new_h);
            return NULL;
        }
        Py_XDECREF(new_h->left);
        new_h->left = new_left;
    } else {
        if (RBNode_is_red(new_h->left)) {
            new_h = RBNode_rotate_right(new_h, edit);
            if (!new_h) return NULL;
        }

        // Check if this is the node to delete
        int eq = PyObject_RichCompareBool(value, new_h->value, Py_EQ);
        if (eq < 0) {
            Py_DECREF(new_h);
            return NULL;
        }

        int key_eq = (SortedVector_compare_keys(sort_key, new_h->sort_key, reverse) == 0);

        if (key_eq && eq && !new_h->right) {
            // Equal-key duplicates can be rotated into the left subtree. Keep
            // that subtree when removing this node instead of dropping every
            // duplicate along with it.
            RBNode *replacement = NULL;
            if (new_h->left) {
                replacement = RBNode_ensure_editable(new_h->left, edit);
                if (!replacement) {
                    Py_DECREF(new_h);
                    return NULL;
                }
            }
            *deleted = 1;
            Py_DECREF(new_h);
            return replacement;
        }

        if (new_h->right && !RBNode_is_red(new_h->right) && !RBNode_is_red(new_h->right->left)) {
            new_h = RBNode_move_red_right(new_h, edit);
            if (!new_h) return NULL;
        }

        // Re-check after rotation
        eq = PyObject_RichCompareBool(value, new_h->value, Py_EQ);
        if (eq < 0) {
            Py_DECREF(new_h);
            return NULL;
        }
        key_eq = (SortedVector_compare_keys(sort_key, new_h->sort_key, reverse) == 0);

        if (key_eq && eq) {
            // Replace with successor
            RBNode *min_node = RBNode_min(new_h->right);

            Py_DECREF(new_h->value);
            new_h->value = min_node->value;
            Py_INCREF(new_h->value);

            Py_DECREF(new_h->sort_key);
            new_h->sort_key = min_node->sort_key;
            Py_INCREF(new_h->sort_key);

            RBNode *new_right = RBNode_delete_min(new_h->right, edit);
            if (new_h->right && !new_right && PyErr_Occurred()) {
                Py_DECREF(new_h);
                return NULL;
            }
            Py_XDECREF(new_h->right);
            new_h->right = new_right;

            *deleted = 1;
        } else {
            RBNode *new_right = RBNode_delete(new_h->right, sort_key, value, reverse, deleted, edit);
            if (new_h->right && !new_right && PyErr_Occurred()) {
                Py_DECREF(new_h);
                return NULL;
            }
            Py_XDECREF(new_h->right);
            new_h->right = new_right;
        }
    }

    return RBNode_balance(new_h, edit);
}

static PyObject *SortedVector_disj(SortedVector *self, PyObject *value) {
    if (self->cnt == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    PyObject *sort_key = SortedVector_get_sort_key(self, value);
    if (!sort_key) return NULL;

    int deleted = 0;
    RBNode *new_root = RBNode_delete(self->root, sort_key, value, self->reverse, &deleted, NULL);
    Py_DECREF(sort_key);

    if (self->root && !new_root && PyErr_Occurred()) return NULL;

    if (!deleted) {
        Py_XDECREF(new_root);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    // Make root black
    if (new_root && new_root->color == RB_RED) {
        new_root->color = RB_BLACK;
    }

    SortedVector *result = PyObject_New(SortedVector, &SortedVectorType);
    if (!result) {
        Py_XDECREF(new_root);
        return NULL;
    }

    result->root = new_root;
    result->cnt = self->cnt - 1;
    result->key_fn = self->key_fn;
    Py_XINCREF(result->key_fn);
    result->reverse = self->reverse;

    return (PyObject *)result;
}

// === SortedVector init ===

static int SortedVector_init(SortedVector *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"", "key", "reverse", NULL};
    PyObject *iterable = NULL;
    PyObject *key_fn = Py_None;
    int reverse = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O$Op", kwlist,
                                     &iterable, &key_fn, &reverse)) {
        return -1;
    }

    self->root = NULL;
    self->cnt = 0;
    self->reverse = reverse;

    if (key_fn != Py_None) {
        self->key_fn = key_fn;
        Py_INCREF(key_fn);
    } else {
        self->key_fn = NULL;
    }

    if (iterable && iterable != Py_None) {
        PyObject *iter = PyObject_GetIter(iterable);
        if (!iter) return -1;

        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            PyObject *new_sv = SortedVector_conj(self, item);
            Py_DECREF(item);
            if (!new_sv) {
                Py_DECREF(iter);
                return -1;
            }

            // Move data from new_sv to self
            Py_XDECREF(self->root);
            self->root = ((SortedVector *)new_sv)->root;
            Py_XINCREF(self->root);
            self->cnt = ((SortedVector *)new_sv)->cnt;
            Py_DECREF(new_sv);
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) return -1;
    }

    return 0;
}

// === SortedVector transient ===

typedef struct TransientSortedVector {
    PyObject_HEAD
    RBNode *root;
    Py_ssize_t cnt;
    PyObject *key_fn;
    int reverse;
    PyObject *id;  // Edit token
} TransientSortedVector;

static PyTypeObject TransientSortedVectorType;

static void TransientSortedVector_dealloc(TransientSortedVector *self) {
    Py_XDECREF(self->root);
    Py_XDECREF(self->key_fn);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void TransientSortedVector_ensure_editable(TransientSortedVector *self) {
    if (!self->id) {
        PyErr_SetString(PyExc_RuntimeError, "TransientSortedVector already made persistent");
    }
}

static PyObject *SortedVector_transient(SortedVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientSortedVector *t = PyObject_New(TransientSortedVector, &TransientSortedVectorType);
    if (!t) return NULL;

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        return NULL;
    }

    t->root = self->root;
    Py_XINCREF(t->root);
    t->cnt = self->cnt;
    t->key_fn = self->key_fn;
    Py_XINCREF(t->key_fn);
    t->reverse = self->reverse;

    return (PyObject *)t;
}

// Get sort key for transient
static PyObject *TransientSortedVector_get_sort_key(TransientSortedVector *self, PyObject *value) {
    if (self->key_fn && self->key_fn != Py_None) {
        return PyObject_CallOneArg(self->key_fn, value);
    }
    Py_INCREF(value);
    return value;
}

static PyObject *TransientSortedVector_conj_mut(TransientSortedVector *self, PyObject *value) {
    TransientSortedVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    PyObject *sort_key = TransientSortedVector_get_sort_key(self, value);
    if (!sort_key) return NULL;

    // Keep each intermediate root isolated. In-place red-black rotations can
    // otherwise alias equal-key subtrees across repeated transient edits.
    RBNode *new_root = RBNode_insert(self->root, value, sort_key, self->reverse, NULL);
    Py_DECREF(sort_key);

    if (!new_root && PyErr_Occurred()) return NULL;

    // Make root black
    if (new_root && new_root->color == RB_RED) {
        new_root->color = RB_BLACK;
    }

    Py_XDECREF(self->root);
    self->root = new_root;
    self->cnt++;

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientSortedVector_disj_mut(TransientSortedVector *self, PyObject *value) {
    TransientSortedVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->cnt == 0) {
        Py_INCREF(self);
        return (PyObject *)self;
    }

    PyObject *sort_key = TransientSortedVector_get_sort_key(self, value);
    if (!sort_key) return NULL;

    int deleted = 0;
    // Use the persistent copy-on-write path so repeated deletes cannot alias
    // equal-key subtrees within the editable tree.
    RBNode *new_root = RBNode_delete(self->root, sort_key, value, self->reverse, &deleted, NULL);
    Py_DECREF(sort_key);

    if (self->root && !new_root && PyErr_Occurred()) return NULL;

    if (deleted) {
        if (new_root && new_root->color == RB_RED) {
            new_root->color = RB_BLACK;
        }
        Py_XDECREF(self->root);
        self->root = new_root;
        self->cnt--;
    } else {
        Py_XDECREF(new_root);
    }

    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *TransientSortedVector_persistent(TransientSortedVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientSortedVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    SortedVector *result = PyObject_New(SortedVector, &SortedVectorType);
    if (!result) return NULL;

    result->root = self->root;
    Py_XINCREF(result->root);
    result->cnt = self->cnt;
    result->key_fn = self->key_fn;
    Py_XINCREF(result->key_fn);
    result->reverse = self->reverse;

    // Invalidate transient
    Py_CLEAR(self->id);

    return (PyObject *)result;
}

static Py_ssize_t TransientSortedVector_length(TransientSortedVector *self) {
    TransientSortedVector_ensure_editable(self);
    if (PyErr_Occurred()) return -1;
    return self->cnt;
}

static PyMethodDef TransientSortedVector_methods[] = {
    {"conj_mut", (PyCFunction)TransientSortedVector_conj_mut, METH_O, "Mutably add element"},
    {"disj_mut", (PyCFunction)TransientSortedVector_disj_mut, METH_O, "Mutably remove element"},
    {"persistent", (PyCFunction)TransientSortedVector_persistent, METH_NOARGS, "Return persistent sorted vector"},
    {NULL}
};

static PySequenceMethods TransientSortedVector_as_sequence = {
    .sq_length = (lenfunc)TransientSortedVector_length,
};

static PyTypeObject TransientSortedVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientSortedVector",
    .tp_doc = "Transient sorted vector for batch operations",
    .tp_basicsize = sizeof(TransientSortedVector),
    .tp_dealloc = (destructor)TransientSortedVector_dealloc,
    .tp_as_sequence = &TransientSortedVector_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = TransientSortedVector_methods,
};

// === SortedVector methods table ===

static PyObject *SortedVector_reduce(SortedVector *self, PyObject *Py_UNUSED(ignored)) {
    // Convert SortedVector to a tuple for the iterable argument
    PyObject *contents = PySequence_Tuple((PyObject *)self);
    if (contents == NULL) {
        return NULL;
    }

    // Build args tuple: (contents_tuple,)
    PyObject *args = PyTuple_Pack(1, contents);
    Py_DECREF(contents);
    if (args == NULL) {
        return NULL;
    }

    // Build kwargs dict with key and reverse
    PyObject *kwargs = PyDict_New();
    if (kwargs == NULL) {
        Py_DECREF(args);
        return NULL;
    }

    // Add key function if present
    if (self->key_fn != NULL) {
        if (PyDict_SetItemString(kwargs, "key", self->key_fn) < 0) {
            Py_DECREF(args);
            Py_DECREF(kwargs);
            return NULL;
        }
    }

    // Add reverse if True
    if (self->reverse) {
        if (PyDict_SetItemString(kwargs, "reverse", Py_True) < 0) {
            Py_DECREF(args);
            Py_DECREF(kwargs);
            return NULL;
        }
    }

    // Get functools.partial to create a callable with kwargs
    PyObject *functools = PyImport_ImportModule("functools");
    if (functools == NULL) {
        Py_DECREF(args);
        Py_DECREF(kwargs);
        return NULL;
    }

    PyObject *partial = PyObject_GetAttrString(functools, "partial");
    Py_DECREF(functools);
    if (partial == NULL) {
        Py_DECREF(args);
        Py_DECREF(kwargs);
        return NULL;
    }

    // Create partial(SortedVector, **kwargs)
    PyObject *partial_args = PyTuple_Pack(1, (PyObject *)Py_TYPE(self));
    if (partial_args == NULL) {
        Py_DECREF(partial);
        Py_DECREF(args);
        Py_DECREF(kwargs);
        return NULL;
    }

    PyObject *reconstructor = PyObject_Call(partial, partial_args, kwargs);
    Py_DECREF(partial);
    Py_DECREF(partial_args);
    Py_DECREF(kwargs);

    if (reconstructor == NULL) {
        Py_DECREF(args);
        return NULL;
    }

    // Return (partial(SortedVector, **kwargs), (contents_tuple,))
    PyObject *result = PyTuple_Pack(2, reconstructor, args);
    Py_DECREF(reconstructor);
    Py_DECREF(args);
    return result;
}

static PyObject *SortedVector_getnewargs_ex(SortedVector *self, PyObject *Py_UNUSED(ignored)) {
    // Convert SortedVector to a tuple for the iterable argument
    PyObject *contents = PySequence_Tuple((PyObject *)self);
    if (contents == NULL) {
        return NULL;
    }

    // Build args tuple: (contents_tuple,)
    PyObject *args = PyTuple_Pack(1, contents);
    Py_DECREF(contents);
    if (args == NULL) {
        return NULL;
    }

    // Build kwargs dict with key and reverse
    PyObject *kwargs = PyDict_New();
    if (kwargs == NULL) {
        Py_DECREF(args);
        return NULL;
    }

    // Add key function if present
    if (self->key_fn != NULL) {
        if (PyDict_SetItemString(kwargs, "key", self->key_fn) < 0) {
            Py_DECREF(args);
            Py_DECREF(kwargs);
            return NULL;
        }
    }

    // Add reverse if True
    if (self->reverse) {
        if (PyDict_SetItemString(kwargs, "reverse", Py_True) < 0) {
            Py_DECREF(args);
            Py_DECREF(kwargs);
            return NULL;
        }
    }

    // Return (args, kwargs) tuple
    PyObject *result = PyTuple_Pack(2, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return result;
}

static PyMethodDef SortedVector_methods[] = {
    {"nth", (PyCFunction)SortedVector_nth, METH_VARARGS, "Get element at index"},
    {"conj", (PyCFunction)SortedVector_conj, METH_O, "Add element maintaining sorted order"},
    {"disj", (PyCFunction)SortedVector_disj, METH_O, "Remove element"},
    {"first", (PyCFunction)SortedVector_first, METH_NOARGS, "Get minimum element"},
    {"last", (PyCFunction)SortedVector_last, METH_NOARGS, "Get maximum element"},
    {"index_of", (PyCFunction)SortedVector_index_of, METH_O, "Find index of element"},
    {"rank", (PyCFunction)SortedVector_rank, METH_O, "Count of elements less than value"},
    {"transient", (PyCFunction)SortedVector_transient, METH_NOARGS, "Get transient version"},
    {"__reduce__", (PyCFunction)SortedVector_reduce, METH_NOARGS, "Pickle support"},
    {"__getnewargs_ex__", (PyCFunction)SortedVector_getnewargs_ex, METH_NOARGS, "Pickle support with keyword args"},
    {"__class_getitem__", (PyCFunction)Generic_class_getitem, METH_O | METH_CLASS,
     "Return a generic alias for type annotations"},
    {NULL}
};

static PySequenceMethods SortedVector_as_sequence = {
    .sq_length = (lenfunc)SortedVector_length,
    .sq_contains = (objobjproc)SortedVector_contains,
};

static PyMappingMethods SortedVector_as_mapping = {
    .mp_length = (lenfunc)SortedVector_length,
    .mp_subscript = (binaryfunc)SortedVector_getitem,
};

static PyTypeObject SortedVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.SortedVector",
    .tp_doc = "Persistent sorted vector (Red-Black Tree)",
    .tp_basicsize = sizeof(SortedVector),
    .tp_dealloc = (destructor)SortedVector_dealloc,
    .tp_repr = (reprfunc)SortedVector_repr,
    .tp_as_sequence = &SortedVector_as_sequence,
    .tp_as_mapping = &SortedVector_as_mapping,
    .tp_hash = (hashfunc)SortedVector_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_iter = (getiterfunc)SortedVector_iter,
    .tp_richcompare = (richcmpfunc)SortedVector_richcompare,
    .tp_methods = SortedVector_methods,
    .tp_new = SortedVector_new,
    .tp_init = (initproc)SortedVector_init,
};

// Empty sorted vector constant
static SortedVector *EMPTY_SORTED_VECTOR = NULL;

// === Module-level functions ===

static PyObject *pds_cons(PyObject *self, PyObject *args) {
    PyObject *first, *rest = Py_None;

    if (!PyArg_ParseTuple(args, "O|O", &first, &rest)) {
        return NULL;
    }

    Cons *c = (Cons *)ConsType.tp_alloc(&ConsType, 0);
    if (!c) return NULL;

    c->first = first;
    Py_INCREF(first);
    c->rest = rest;
    Py_INCREF(rest);
    c->hash = 0;
    c->hash_computed = 0;

    return (PyObject *)c;
}

static PyObject *pds_vec(PyObject *self, PyObject *args) {
    Py_ssize_t n = PyTuple_Size(args);

    if (n == 0) {
        Py_INCREF(EMPTY_VECTOR);
        return (PyObject *)EMPTY_VECTOR;
    }

    // Check for single iterable argument
    if (n == 1) {
        PyObject *arg = PyTuple_GET_ITEM(args, 0);
        if (PyIter_Check(arg) || PyObject_TypeCheck(arg, &SortedVectorType) ||
            (PySequence_Check(arg) && !PyUnicode_Check(arg) &&
            !PyObject_TypeCheck(arg, &VectorType) && !PyObject_TypeCheck(arg, &MapType))) {
            // Single iterable - expand it
            PyObject *iter = PyObject_GetIter(arg);
            if (!iter) {
                PyErr_Clear();
                // Not iterable, treat as single element
            } else {
                TransientVector *t = (TransientVector *)Vector_transient(EMPTY_VECTOR, NULL);
                if (!t) {
                    Py_DECREF(iter);
                    return NULL;
                }

                PyObject *item;
                while ((item = PyIter_Next(iter)) != NULL) {
                    PyObject *result = TransientVector_conj_mut(t, item);
                    Py_DECREF(item);
                    if (!result) {
                        Py_DECREF(iter);
                        Py_DECREF(t);
                        return NULL;
                    }
                    Py_DECREF(result);
                }
                Py_DECREF(iter);

                if (PyErr_Occurred()) {
                    Py_DECREF(t);
                    return NULL;
                }

                return TransientVector_persistent(t, NULL);
            }
        }
    }

    // Multiple arguments or single non-iterable
    TransientVector *t = (TransientVector *)Vector_transient(EMPTY_VECTOR, NULL);
    if (!t) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        PyObject *result = TransientVector_conj_mut(t, item);
        if (!result) {
            Py_DECREF(t);
            return NULL;
        }
        Py_DECREF(result);
    }

    return TransientVector_persistent(t, NULL);
}

static PyObject *pds_set(PyObject *self, PyObject *args) {
    PyObject *iterable = NULL;

    if (!PyArg_ParseTuple(args, "|O", &iterable)) {
        return NULL;
    }

    if (iterable == NULL) {
        Py_INCREF(EMPTY_SET);
        return (PyObject *)EMPTY_SET;
    }

    PyObject *iter = PyObject_GetIter(iterable);
    if (!iter) return NULL;

    // Use transient for efficient building
    TransientSet *t = PyObject_New(TransientSet, &TransientSetType);
    if (!t) {
        Py_DECREF(iter);
        return NULL;
    }

    t->id = PyObject_New(PyObject, &PdsSentinelType);
    if (!t->id) {
        Py_DECREF(t);
        Py_DECREF(iter);
        return NULL;
    }

    t->cnt = 0;
    t->root = NULL;

    PyObject *key;
    while ((key = PyIter_Next(iter)) != NULL) {
        PyObject *result = TransientSet_conj_mut(t, key);
        Py_DECREF(key);
        if (!result) {
            Py_DECREF(t);
            Py_DECREF(iter);
            return NULL;
        }
        Py_DECREF(result);
    }
    Py_DECREF(iter);

    if (PyErr_Occurred()) {
        Py_DECREF(t);
        return NULL;
    }

    PyObject *result = TransientSet_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}

static PyObject *pds_hash_map(PyObject *self, PyObject *args) {
    Py_ssize_t n = PyTuple_Size(args);

    if (n % 2 != 0) {
        PyErr_SetString(PyExc_ValueError, "hash_map requires an even number of arguments");
        return NULL;
    }

    if (n == 0) {
        Py_INCREF(EMPTY_MAP);
        return (PyObject *)EMPTY_MAP;
    }

    TransientMap *t = (TransientMap *)Map_transient(EMPTY_MAP, NULL);
    if (!t) return NULL;

    for (Py_ssize_t i = 0; i < n; i += 2) {
        PyObject *key = PyTuple_GET_ITEM(args, i);
        PyObject *val = PyTuple_GET_ITEM(args, i + 1);

        PyObject *kv_args = PyTuple_Pack(2, key, val);
        if (!kv_args) {
            Py_DECREF(t);
            return NULL;
        }

        PyObject *result = TransientMap_assoc_mut(t, kv_args);
        Py_DECREF(kv_args);
        if (!result) {
            Py_DECREF(t);
            return NULL;
        }
        Py_DECREF(result);
    }

    return TransientMap_persistent(t, NULL);
}

// =============================================================================
// FACTORY FUNCTIONS FOR TYPE-SPECIALIZED VECTORS
// =============================================================================

static PyObject *pds_vec_f64(PyObject *self, PyObject *args) {
    Py_ssize_t n = PyTuple_Size(args);

    if (n == 0) {
        Py_INCREF(EMPTY_DOUBLE_VECTOR);
        return (PyObject *)EMPTY_DOUBLE_VECTOR;
    }

    // Build using transient for O(1) amortized appends
    TransientDoubleVector *t = (TransientDoubleVector *)DoubleVector_transient(EMPTY_DOUBLE_VECTOR, NULL);
    if (!t) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        double val = PyFloat_AsDouble(item);

        if (val == -1.0 && PyErr_Occurred()) {
            Py_DECREF(t);
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError,
                "vec_f64 argument %zd must be a number, got %s",
                i, Py_TYPE(item)->tp_name);
            return NULL;
        }

        // Use raw function to avoid boxing/unboxing overhead
        if (TransientDoubleVector_conj_mut_raw(t, val) < 0) {
            Py_DECREF(t);
            return NULL;
        }
    }

    return TransientDoubleVector_persistent(t, NULL);
}

static PyObject *pds_vec_i64(PyObject *self, PyObject *args) {
    Py_ssize_t n = PyTuple_Size(args);

    if (n == 0) {
        Py_INCREF(EMPTY_LONG_VECTOR);
        return (PyObject *)EMPTY_LONG_VECTOR;
    }

    // Build using transient for O(1) amortized appends
    TransientIntVector *t = (TransientIntVector *)IntVector_transient(EMPTY_LONG_VECTOR, NULL);
    if (!t) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        int64_t val = PyLong_AsLongLong(item);

        if (val == -1 && PyErr_Occurred()) {
            Py_DECREF(t);
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError,
                "vec_i64 argument %zd must be an integer, got %s",
                i, Py_TYPE(item)->tp_name);
            return NULL;
        }

        // Use raw function to avoid boxing/unboxing overhead
        if (TransientIntVector_conj_mut_raw(t, val) < 0) {
            Py_DECREF(t);
            return NULL;
        }
    }

    return TransientIntVector_persistent(t, NULL);
}

static PyObject *pds_sorted_vec(PyObject *self, PyObject *args, PyObject *kwargs) {
    static char *kwlist[] = {"", "key", "reverse", NULL};
    PyObject *iterable = NULL;
    PyObject *key_fn = Py_None;
    int reverse = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O$Op", kwlist,
                                     &iterable, &key_fn, &reverse)) {
        return NULL;
    }

    SortedVector *sv = (SortedVector *)SortedVector_new(&SortedVectorType, NULL, NULL);
    if (!sv) return NULL;

    sv->reverse = reverse;
    if (key_fn != Py_None) {
        sv->key_fn = key_fn;
        Py_INCREF(key_fn);
    }

    if (iterable && iterable != Py_None) {
        PyObject *iter = PyObject_GetIter(iterable);
        if (!iter) {
            Py_DECREF(sv);
            return NULL;
        }

        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            PyObject *new_sv = SortedVector_conj(sv, item);
            Py_DECREF(item);
            if (!new_sv) {
                Py_DECREF(iter);
                Py_DECREF(sv);
                return NULL;
            }
            Py_DECREF(sv);
            sv = (SortedVector *)new_sv;
        }
        Py_DECREF(iter);

        if (PyErr_Occurred()) {
            Py_DECREF(sv);
            return NULL;
        }
    }

    return (PyObject *)sv;
}

static PyMethodDef pds_methods[] = {
    {"cons", pds_cons, METH_VARARGS, "Create a cons cell"},
    {"vec", pds_vec, METH_VARARGS, "Create a persistent vector"},
    {"vec_f64", pds_vec_f64, METH_VARARGS, "Create a persistent vector of float64"},
    {"vec_i64", pds_vec_i64, METH_VARARGS, "Create a persistent vector of int64"},
    {"hash_map", pds_hash_map, METH_VARARGS, "Create a persistent map from key-value pairs"},
    {"hash_set", pds_set, METH_VARARGS, "Create a persistent set from an iterable"},
    {"sorted_vec", (PyCFunction)pds_sorted_vec, METH_VARARGS | METH_KEYWORDS, "Create a persistent sorted vector"},
    {NULL, NULL, 0, NULL}
};

// =============================================================================
// MODULE STATE MANAGEMENT (GC integration)
// =============================================================================

static int pds_traverse(PyObject *m, visitproc visit, void *arg) {
    PdsState *st = pds_get_state(m);
    if (st == NULL) return 0;

#if !PDS_SINGLETONS_ARE_IMMORTAL
    Py_VISIT(st->_MISSING);
    Py_VISIT(st->EMPTY_VECTOR);
    Py_VISIT(st->EMPTY_DOUBLE_VECTOR);
    Py_VISIT(st->EMPTY_LONG_VECTOR);
    Py_VISIT(st->EMPTY_MAP);
    Py_VISIT(st->EMPTY_SET);
    Py_VISIT(st->EMPTY_SORTED_VECTOR);
    Py_VISIT(st->EMPTY_NODE);
    Py_VISIT(st->EMPTY_DOUBLE_NODE);
    Py_VISIT(st->EMPTY_LONG_NODE);
    Py_VISIT(st->EMPTY_BIN);
#endif

    return 0;
}

static int pds_clear(PyObject *m) {
    PdsState *st = pds_get_state(m);
    if (st == NULL) return 0;

#if !PDS_SINGLETONS_ARE_IMMORTAL
    Py_CLEAR(st->_MISSING);
    Py_CLEAR(st->EMPTY_VECTOR);
    Py_CLEAR(st->EMPTY_DOUBLE_VECTOR);
    Py_CLEAR(st->EMPTY_LONG_VECTOR);
    Py_CLEAR(st->EMPTY_MAP);
    Py_CLEAR(st->EMPTY_SET);
    Py_CLEAR(st->EMPTY_SORTED_VECTOR);
    Py_CLEAR(st->EMPTY_NODE);
    Py_CLEAR(st->EMPTY_DOUBLE_NODE);
    Py_CLEAR(st->EMPTY_LONG_NODE);
    Py_CLEAR(st->EMPTY_BIN);
#else
    st->_MISSING = NULL;
    st->EMPTY_VECTOR = NULL;
    st->EMPTY_DOUBLE_VECTOR = NULL;
    st->EMPTY_LONG_VECTOR = NULL;
    st->EMPTY_MAP = NULL;
    st->EMPTY_SET = NULL;
    st->EMPTY_SORTED_VECTOR = NULL;
    st->EMPTY_NODE = NULL;
    st->EMPTY_DOUBLE_NODE = NULL;
    st->EMPTY_LONG_NODE = NULL;
    st->EMPTY_BIN = NULL;
#endif

    return 0;
}

static void pds_free(void *m) {
#if !PDS_SINGLETONS_ARE_IMMORTAL
    pds_clear((PyObject *)m);
#endif
}

static int pds_exec(PyObject *m);

static PyModuleDef_Slot pds_slots[] = {
    {Py_mod_exec, pds_exec},
#if PY_VERSION_HEX >= 0x030D0000
    // Python 3.13+: Declare this module as free-threading safe
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}
};

static struct PyModuleDef pdsmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "spork_pds",
    .m_doc = "Persistent Data Structures - C implementation",
    .m_size = sizeof(PdsState),
    .m_methods = pds_methods,
    .m_slots = pds_slots,
    .m_traverse = pds_traverse,
    .m_clear = pds_clear,
    .m_free = pds_free,
};

static int pds_exec(PyObject *m) {
    PdsState *st = pds_get_state(m);
    if (st == NULL) {
        return -1;
    }

    // Initialize sentinel type first (used for identity objects)
    if (PyType_Ready(&PdsSentinelType) < 0) return -1;

    // Initialize types
    if (PyType_Ready(&ConsType) < 0) return -1;
    if (PyType_Ready(&ConsIteratorType) < 0) return -1;
    if (PyType_Ready(&VectorNodeType) < 0) return -1;
    if (PyType_Ready(&VectorType) < 0) return -1;
    if (PyType_Ready(&VectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientVectorType) < 0) return -1;
    if (PyType_Ready(&TransientVectorIteratorType) < 0) return -1;

    // Initialize SortedVector types
    if (PyType_Ready(&RBNodeType) < 0) return -1;
    if (PyType_Ready(&SortedVectorType) < 0) return -1;
    if (PyType_Ready(&SortedVectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientSortedVectorType) < 0) return -1;

    // Initialize type-specialized vector types
    if (PyType_Ready(&DoubleVectorNodeType) < 0) return -1;
    if (PyType_Ready(&DoubleVectorType) < 0) return -1;
    if (PyType_Ready(&DoubleVectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientDoubleVectorType) < 0) return -1;
    if (PyType_Ready(&IntVectorNodeType) < 0) return -1;
    if (PyType_Ready(&IntVectorType) < 0) return -1;
    if (PyType_Ready(&IntVectorIteratorType) < 0) return -1;
    if (PyType_Ready(&TransientIntVectorType) < 0) return -1;

    if (PyType_Ready(&BitmapIndexedNodeType) < 0) return -1;
    if (PyType_Ready(&ArrayNodeType) < 0) return -1;
    if (PyType_Ready(&HashCollisionNodeType) < 0) return -1;
    if (PyType_Ready(&BitmapIndexedNodeIteratorType) < 0) return -1;
    if (PyType_Ready(&ArrayNodeIteratorType) < 0) return -1;
    if (PyType_Ready(&HashCollisionNodeIteratorType) < 0) return -1;
    if (PyType_Ready(&MapType) < 0) return -1;
    if (PyType_Ready(&TransientMapType) < 0) return -1;
    if (PyType_Ready(&SetType) < 0) return -1;
    if (PyType_Ready(&TransientSetType) < 0) return -1;
    if (PyType_Ready(&SetIteratorType) < 0) return -1;

    // Create singletons only once to ensure sub-interpreter safety.
    // If already initialized, reuse the existing global singletons.
    if (_singletons_initialized) {
        // Reuse existing singletons - just copy globals to module state
        st->_MISSING = _MISSING;
        st->EMPTY_NODE = (PyObject *)EMPTY_NODE;
        st->EMPTY_VECTOR = (PyObject *)EMPTY_VECTOR;
        st->EMPTY_DOUBLE_NODE = (PyObject *)EMPTY_DOUBLE_NODE;
        st->EMPTY_DOUBLE_VECTOR = (PyObject *)EMPTY_DOUBLE_VECTOR;
        st->EMPTY_LONG_NODE = (PyObject *)EMPTY_LONG_NODE;
        st->EMPTY_LONG_VECTOR = (PyObject *)EMPTY_LONG_VECTOR;
        st->EMPTY_BIN = (PyObject *)EMPTY_BIN;
        st->EMPTY_MAP = (PyObject *)EMPTY_MAP;
        st->EMPTY_SET = (PyObject *)EMPTY_SET;
        st->EMPTY_SORTED_VECTOR = (PyObject *)EMPTY_SORTED_VECTOR;
    } else {
        // First initialization - create all singletons

        st->_MISSING = PyObject_New(PyObject, &PdsSentinelType);
        if (!st->_MISSING) return -1;

        // Create empty node
        st->EMPTY_NODE = (PyObject *)VectorNode_create(NULL);
        if (!st->EMPTY_NODE) return -1;

        // Create empty vector
        st->EMPTY_VECTOR = (PyObject *)Vector_create(0, BITS, (VectorNode *)st->EMPTY_NODE, NULL, NULL);
        if (!st->EMPTY_VECTOR) return -1;

        // Create empty double vector node and vector
        st->EMPTY_DOUBLE_NODE = (PyObject *)DoubleVectorNode_create(NULL);
        if (!st->EMPTY_DOUBLE_NODE) return -1;

        st->EMPTY_DOUBLE_VECTOR = (PyObject *)DoubleVector_create(0, BITS, (DoubleVectorNode *)st->EMPTY_DOUBLE_NODE, NULL, 0, NULL);
        if (!st->EMPTY_DOUBLE_VECTOR) return -1;

        // Create empty long vector node and vector
        st->EMPTY_LONG_NODE = (PyObject *)IntVectorNode_create(NULL);
        if (!st->EMPTY_LONG_NODE) return -1;

        st->EMPTY_LONG_VECTOR = (PyObject *)IntVector_create(0, BITS, (IntVectorNode *)st->EMPTY_LONG_NODE, NULL, 0, NULL);
        if (!st->EMPTY_LONG_VECTOR) return -1;

        // Create empty bitmap indexed node
        st->EMPTY_BIN = (PyObject *)BitmapIndexedNode_create(0, NULL, NULL);
        if (!st->EMPTY_BIN) return -1;

        // Create empty map
        st->EMPTY_MAP = (PyObject *)Map_create(0, NULL, NULL);
        if (!st->EMPTY_MAP) return -1;

        // Create empty set
        st->EMPTY_SET = (PyObject *)Set_create(0, NULL, NULL);
        if (!st->EMPTY_SET) return -1;

        // Create empty sorted vector
        st->EMPTY_SORTED_VECTOR = (PyObject *)PyObject_New(SortedVector, &SortedVectorType);
        if (!st->EMPTY_SORTED_VECTOR) return -1;
        {
            SortedVector *sv = (SortedVector *)st->EMPTY_SORTED_VECTOR;
            sv->root = NULL;
            sv->cnt = 0;
            sv->key_fn = NULL;
            sv->reverse = 0;
        }

        // Immortalize singletons for Python 3.12+ to prevent refcount contention
        // in multi-threaded code. Immortal objects don't have their refcounts modified.
        PDS_SET_IMMORTAL(st->_MISSING);
        PDS_SET_IMMORTAL(st->EMPTY_NODE);
        PDS_SET_IMMORTAL(st->EMPTY_VECTOR);
        PDS_SET_IMMORTAL(st->EMPTY_DOUBLE_NODE);
        PDS_SET_IMMORTAL(st->EMPTY_DOUBLE_VECTOR);
        PDS_SET_IMMORTAL(st->EMPTY_LONG_NODE);
        PDS_SET_IMMORTAL(st->EMPTY_LONG_VECTOR);
        PDS_SET_IMMORTAL(st->EMPTY_BIN);
        PDS_SET_IMMORTAL(st->EMPTY_MAP);
        PDS_SET_IMMORTAL(st->EMPTY_SET);
        PDS_SET_IMMORTAL(st->EMPTY_SORTED_VECTOR);

        // Update global aliases for backward compatibility with existing code
        // These are set once and never change, ensuring sub-interpreter safety
        _MISSING = st->_MISSING;
        EMPTY_NODE = (VectorNode *)st->EMPTY_NODE;
        EMPTY_VECTOR = (Vector *)st->EMPTY_VECTOR;
        EMPTY_DOUBLE_NODE = (DoubleVectorNode *)st->EMPTY_DOUBLE_NODE;
        EMPTY_DOUBLE_VECTOR = (DoubleVector *)st->EMPTY_DOUBLE_VECTOR;
        EMPTY_LONG_NODE = (IntVectorNode *)st->EMPTY_LONG_NODE;
        EMPTY_LONG_VECTOR = (IntVector *)st->EMPTY_LONG_VECTOR;
        EMPTY_BIN = (BitmapIndexedNode *)st->EMPTY_BIN;
        EMPTY_MAP = (Map *)st->EMPTY_MAP;
        EMPTY_SET = (Set *)st->EMPTY_SET;
        EMPTY_SORTED_VECTOR = (SortedVector *)st->EMPTY_SORTED_VECTOR;

        // Mark as initialized
        _singletons_initialized = 1;
    }

    // Add types to module
    Py_INCREF(&ConsType);
    if (PyModule_AddObject(m, "Cons", (PyObject *)&ConsType) < 0) {
        Py_DECREF(&ConsType);
        return -1;
    }

    Py_INCREF(&VectorType);
    if (PyModule_AddObject(m, "Vector", (PyObject *)&VectorType) < 0) {
        Py_DECREF(&VectorType);
        return -1;
    }

    Py_INCREF(&TransientVectorType);
    if (PyModule_AddObject(m, "TransientVector", (PyObject *)&TransientVectorType) < 0) {
        Py_DECREF(&TransientVectorType);
        return -1;
    }

    Py_INCREF(&MapType);
    if (PyModule_AddObject(m, "Map", (PyObject *)&MapType) < 0) {
        Py_DECREF(&MapType);
        return -1;
    }

    Py_INCREF(&TransientMapType);
    if (PyModule_AddObject(m, "TransientMap", (PyObject *)&TransientMapType) < 0) {
        Py_DECREF(&TransientMapType);
        return -1;
    }

    Py_INCREF(&SetType);
    if (PyModule_AddObject(m, "Set", (PyObject *)&SetType) < 0) {
        Py_DECREF(&SetType);
        return -1;
    }

    Py_INCREF(&TransientSetType);
    if (PyModule_AddObject(m, "TransientSet", (PyObject *)&TransientSetType) < 0) {
        Py_DECREF(&TransientSetType);
        return -1;
    }

    // Add empty instances
    Py_INCREF(st->EMPTY_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_VECTOR", st->EMPTY_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_VECTOR);
        return -1;
    }

    Py_INCREF(st->EMPTY_MAP);
    if (PyModule_AddObject(m, "EMPTY_MAP", st->EMPTY_MAP) < 0) {
        Py_DECREF(st->EMPTY_MAP);
        return -1;
    }

    Py_INCREF(st->EMPTY_SET);
    if (PyModule_AddObject(m, "EMPTY_SET", st->EMPTY_SET) < 0) {
        Py_DECREF(st->EMPTY_SET);
        return -1;
    }

    // Add SortedVector types
    Py_INCREF(&SortedVectorType);
    if (PyModule_AddObject(m, "SortedVector", (PyObject *)&SortedVectorType) < 0) {
        Py_DECREF(&SortedVectorType);
        return -1;
    }

    Py_INCREF(&TransientSortedVectorType);
    if (PyModule_AddObject(m, "TransientSortedVector", (PyObject *)&TransientSortedVectorType) < 0) {
        Py_DECREF(&TransientSortedVectorType);
        return -1;
    }

    Py_INCREF(st->EMPTY_SORTED_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_SORTED_VECTOR", st->EMPTY_SORTED_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_SORTED_VECTOR);
        return -1;
    }

    // Add type-specialized vector types
    Py_INCREF(&DoubleVectorType);
    if (PyModule_AddObject(m, "DoubleVector", (PyObject *)&DoubleVectorType) < 0) {
        Py_DECREF(&DoubleVectorType);
        return -1;
    }

    Py_INCREF(&IntVectorType);
    if (PyModule_AddObject(m, "IntVector", (PyObject *)&IntVectorType) < 0) {
        Py_DECREF(&IntVectorType);
        return -1;
    }

    Py_INCREF(&TransientDoubleVectorType);
    if (PyModule_AddObject(m, "TransientDoubleVector", (PyObject *)&TransientDoubleVectorType) < 0) {
        Py_DECREF(&TransientDoubleVectorType);
        return -1;
    }

    Py_INCREF(&TransientIntVectorType);
    if (PyModule_AddObject(m, "TransientIntVector", (PyObject *)&TransientIntVectorType) < 0) {
        Py_DECREF(&TransientIntVectorType);
        return -1;
    }

    // Add empty specialized vector instances
    Py_INCREF(st->EMPTY_DOUBLE_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_DOUBLE_VECTOR", st->EMPTY_DOUBLE_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_DOUBLE_VECTOR);
        return -1;
    }

    Py_INCREF(st->EMPTY_LONG_VECTOR);
    if (PyModule_AddObject(m, "EMPTY_LONG_VECTOR", st->EMPTY_LONG_VECTOR) < 0) {
        Py_DECREF(st->EMPTY_LONG_VECTOR);
        return -1;
    }

    // Register types with collections.abc ABCs
    // This enables isinstance() checks for protocol compatibility
    PyObject *collections_abc = PyImport_ImportModule("collections.abc");
    if (collections_abc) {
        PyObject *result;

        // Register immutable Sequence types
        PyObject *sequence_abc = PyObject_GetAttrString(collections_abc, "Sequence");
        if (sequence_abc) {
            // Register Vector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &VectorType);
            Py_XDECREF(result);

            // Register DoubleVector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &DoubleVectorType);
            Py_XDECREF(result);

            // Register IntVector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &IntVectorType);
            Py_XDECREF(result);

            // Register Cons as well
            result = PyObject_CallMethod(sequence_abc, "register", "O", &ConsType);
            Py_XDECREF(result);

            // Register SortedVector
            result = PyObject_CallMethod(sequence_abc, "register", "O", &SortedVectorType);
            Py_XDECREF(result);

            Py_DECREF(sequence_abc);
        }

        // Register immutable Mapping type
        PyObject *mapping_abc = PyObject_GetAttrString(collections_abc, "Mapping");
        if (mapping_abc) {
            result = PyObject_CallMethod(mapping_abc, "register", "O", &MapType);
            Py_XDECREF(result);
            Py_DECREF(mapping_abc);
        }

        // Register immutable Set type
        PyObject *set_abc = PyObject_GetAttrString(collections_abc, "Set");
        if (set_abc) {
            result = PyObject_CallMethod(set_abc, "register", "O", &SetType);
            Py_XDECREF(result);
            Py_DECREF(set_abc);
        }

        // Register TransientVector as MutableSequence
        PyObject *mutable_sequence_abc = PyObject_GetAttrString(collections_abc, "MutableSequence");
        if (mutable_sequence_abc) {
            result = PyObject_CallMethod(mutable_sequence_abc, "register", "O", &TransientVectorType);
            Py_XDECREF(result);
            Py_DECREF(mutable_sequence_abc);
        }

        // Register TransientMap as MutableMapping
        PyObject *mutable_mapping_abc = PyObject_GetAttrString(collections_abc, "MutableMapping");
        if (mutable_mapping_abc) {
            result = PyObject_CallMethod(mutable_mapping_abc, "register", "O", &TransientMapType);
            Py_XDECREF(result);
            Py_DECREF(mutable_mapping_abc);
        }

        // Register TransientSet as MutableSet
        PyObject *mutable_set_abc = PyObject_GetAttrString(collections_abc, "MutableSet");
        if (mutable_set_abc) {
            result = PyObject_CallMethod(mutable_set_abc, "register", "O", &TransientSetType);
            Py_XDECREF(result);
            Py_DECREF(mutable_set_abc);
        }

        Py_DECREF(collections_abc);
    }
    // Clear any import errors - ABC registration is optional
    PyErr_Clear();

    return 0;
}

PyMODINIT_FUNC PyInit_spork_pds(void)
{
    return PyModuleDef_Init(&pdsmodule);
}
