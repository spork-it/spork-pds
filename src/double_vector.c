#include "pds_internal.h"

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
    uint32_t valid_mask;  // Bitmask of which slots are valid
    int is_leaf;
    PyObject *transient_id;
} DoubleVectorNode;

PyTypeObject DoubleVectorNodeType;

static void DoubleVectorNode_dealloc(DoubleVectorNode *self) {
    if (!self->is_leaf) {
        for (int i = 0; i < WIDTH; i++) {
            if (self->valid_mask & (1U << i)) {
                Py_XDECREF(self->data.children[i]);
            }
        }
    }
    Py_XDECREF(self->transient_id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static DoubleVectorNode *
DoubleVectorNode_create_kind(PyObject *transient_id, int is_leaf) {
    DoubleVectorNode *node = (DoubleVectorNode *)DoubleVectorNodeType.tp_alloc(
        &DoubleVectorNodeType, 0);
    if (!node) return NULL;

    memset(&node->data, 0, sizeof(node->data));
    node->valid_mask = 0;
    node->is_leaf = is_leaf;
    node->transient_id = transient_id;
    Py_XINCREF(transient_id);
    return node;
}

DoubleVectorNode *DoubleVectorNode_create(PyObject *transient_id) {
    return DoubleVectorNode_create_kind(transient_id, 0);
}

static DoubleVectorNode *DoubleVectorNode_create_leaf(PyObject *transient_id) {
    return DoubleVectorNode_create_kind(transient_id, 1);
}

static DoubleVectorNode *DoubleVectorNode_clone(DoubleVectorNode *self, PyObject *transient_id) {
    DoubleVectorNode *node =
        DoubleVectorNode_create_kind(transient_id, self->is_leaf);
    if (!node) return NULL;

    memcpy(&node->data, &self->data, sizeof(self->data));
    node->valid_mask = self->valid_mask;
    if (!self->is_leaf) {
        for (int i = 0; i < WIDTH; i++) {
            if (self->valid_mask & (1U << i)) {
                Py_XINCREF(node->data.children[i]);
            }
        }
    }
    return node;
}

static int DoubleVectorNode_is_editable(DoubleVectorNode *self, PyObject *transient_id) {
    return transient_id != NULL && self->transient_id == transient_id;
}

PyTypeObject DoubleVectorNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.DoubleVectorNode",
    .tp_basicsize = sizeof(DoubleVectorNode),
    .tp_dealloc = (destructor)DoubleVectorNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_alloc = PyType_GenericAlloc,
};

// Global empty double node
DoubleVectorNode *EMPTY_DOUBLE_NODE = NULL;

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

PyTypeObject DoubleVectorType;
DoubleVector *EMPTY_DOUBLE_VECTOR = NULL;

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

DoubleVector *DoubleVector_create(Py_ssize_t cnt, int shift, DoubleVectorNode *root,
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
        if (child != NULL && (parent->valid_mask & (1U << subidx))) {
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
    if (ret->valid_mask & (1U << subidx)) {
        Py_XDECREF(ret->data.children[subidx]);
    }
    ret->data.children[subidx] = node_to_insert;
    ret->valid_mask |= (1U << subidx);
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
    DoubleVectorNode *tail_node = DoubleVectorNode_create_leaf(transient_id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1U << i);
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
    Py_hash_t cached_hash = 0;
    int hash_computed;

    PDS_BEGIN_CRITICAL_SECTION(self);
    hash_computed = self->hash_computed;
    if (hash_computed) {
        cached_hash = self->hash;
    }
    PDS_END_CRITICAL_SECTION();

    if (hash_computed) {
        return cached_hash;
    }

    Py_uhash_t h = 0;
    for (Py_ssize_t i = 0; i < self->cnt; i++) {
        double val = DoubleVector_nth_raw(self, i);
        // Hash the double bytes
        Py_hash_t item_hash = _Py_HashDouble((PyObject *)self, val);
        if (item_hash == -1 && PyErr_Occurred()) {
            return -1;
        }
        h = (Py_uhash_t)31 * h + (Py_uhash_t)item_hash;
    }

    Py_hash_t computed_hash = pds_finalize_hash(h);
    PDS_BEGIN_CRITICAL_SECTION(self);
    if (!self->hash_computed) {
        self->hash = computed_hash;
        self->hash_computed = 1;
    }
    cached_hash = self->hash;
    PDS_END_CRITICAL_SECTION();
    return cached_hash;
}

// Buffer Protocol Implementation for DoubleVector
static int DoubleVector_flatten(DoubleVector *self) {
    int cache_initialized;
    PDS_BEGIN_CRITICAL_SECTION(self);
    cache_initialized = self->flat_buffer_cache != NULL;
    PDS_END_CRITICAL_SECTION();

    if (cache_initialized) {
        return 0;
    }

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

    PDS_BEGIN_CRITICAL_SECTION(self);
    if (self->flat_buffer_cache == NULL) {
        self->flat_buffer_cache = buffer;
        buffer = NULL;
    }
    PDS_END_CRITICAL_SECTION();
    free(buffer);
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
    int busy;
} DoubleVectorIterator;

PyTypeObject DoubleVectorIteratorType;

static void DoubleVectorIterator_dealloc(DoubleVectorIterator *self) {
    Py_XDECREF(self->vec);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DoubleVectorIterator_next_impl(DoubleVectorIterator *self) {
    if (self->index >= self->vec->cnt) {
        return NULL;  // StopIteration
    }

    double val = DoubleVector_nth_raw(self->vec, self->index);
    if (PyErr_Occurred()) return NULL;
    self->index++;
    return PyFloat_FromDouble(val);
}

static PyObject *DoubleVectorIterator_next(DoubleVectorIterator *self) {
    PyObject *result = NULL;

    PDS_BEGIN_CRITICAL_SECTION(self);
    if (self->busy) {
        PyErr_SetString(PyExc_RuntimeError, PDS_ITERATOR_BUSY_ERROR);
    } else {
        self->busy = 1;
        result = DoubleVectorIterator_next_impl(self);
        self->busy = 0;
    }
    PDS_END_CRITICAL_SECTION();

    return result;
}

PyTypeObject DoubleVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.DoubleVectorIterator",
    .tp_basicsize = sizeof(DoubleVectorIterator),
    .tp_dealloc = (destructor)DoubleVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter,
    .tp_iternext = (iternextfunc)DoubleVectorIterator_next,
};

static PyObject *DoubleVector_iter(DoubleVector *self) {
    DoubleVectorIterator *it =
        (DoubleVectorIterator *)DoubleVectorIteratorType.tp_alloc(
            &DoubleVectorIteratorType, 0);
    if (!it) return NULL;
    it->vec = self;
    Py_INCREF(self);
    it->index = 0;
    it->busy = 0;
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
    uint64_t owner_thread_id;
} TransientDoubleVector;

PyTypeObject TransientDoubleVectorType;

static void TransientDoubleVector_dealloc(TransientDoubleVector *self) {
    Py_XDECREF(self->root);
    if (self->tail) free(self->tail);
    Py_XDECREF(self->id);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void TransientDoubleVector_ensure_editable(TransientDoubleVector *self) {
    if (pds_check_transient_owner(self->owner_thread_id) < 0) {
        return;
    }
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
        if (child != NULL && (parent->valid_mask & (1U << subidx))) {
            node_to_insert = TransientDoubleVector_push_tail(self, level - BITS, child, tail_node);
        } else {
            node_to_insert = TransientDoubleVector_new_path(self, level - BITS, tail_node);
        }
        if (!node_to_insert) {
            Py_DECREF(ret);
            return NULL;
        }
    }

    if (ret->valid_mask & (1U << subidx)) {
        Py_XDECREF(ret->data.children[subidx]);
    }
    ret->data.children[subidx] = node_to_insert;
    ret->valid_mask |= (1U << subidx);
    return ret;
}

/*
 * Factory-only raw append.  The caller owns an unpublished, owner-initialized
 * transient, so the Python-visible owner/editability check is not repeated.
 */
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
    DoubleVectorNode *tail_node = DoubleVectorNode_create_leaf(self->id);
    if (!tail_node) return -1;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1U << i);
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
    DoubleVectorNode *tail_node = DoubleVectorNode_create_leaf(self->id);
    if (!tail_node) return NULL;

    for (Py_ssize_t i = 0; i < self->tail_len && i < WIDTH; i++) {
        tail_node->data.values[i] = self->tail[i];
        tail_node->valid_mask |= (1U << i);
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

PyTypeObject TransientDoubleVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientDoubleVector",
    .tp_doc = "Transient double vector for batch operations",
    .tp_basicsize = sizeof(TransientDoubleVector),
    .tp_dealloc = (destructor)TransientDoubleVector_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = TransientDoubleVector_methods,
};

static PyObject *DoubleVector_transient(DoubleVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientDoubleVector *t =
        (TransientDoubleVector *)TransientDoubleVectorType.tp_alloc(
            &TransientDoubleVectorType, 0);
    if (!t) return NULL;

    t->owner_thread_id = pds_current_thread_state_id();
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
                    self->tail = NULL;
                    self->tail_len = 0;
                    self->tail_cap = 0;
                    self->cnt = nv->cnt;
                    self->shift = nv->shift;
                    self->root = nv->root;
                    Py_INCREF(self->root);
                    if (nv->tail && nv->tail_len > 0) {
                        self->tail = (double *)malloc(nv->tail_len * sizeof(double));
                        if (!self->tail) {
                            Py_DECREF(new_vec);
                            Py_DECREF(iter);
                            PyErr_NoMemory();
                            return -1;
                        }
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
        self->tail = NULL;
        self->tail_len = 0;
        self->tail_cap = 0;
        self->cnt = nv->cnt;
        self->shift = nv->shift;
        self->root = nv->root;
        Py_INCREF(self->root);
        if (nv->tail && nv->tail_len > 0) {
            self->tail = (double *)malloc(nv->tail_len * sizeof(double));
            if (!self->tail) {
                Py_DECREF(new_vec);
                PyErr_NoMemory();
                return -1;
            }
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

PyTypeObject DoubleVectorType = {
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

PyObject *pds_vec_f64(PyObject *self, PyObject *args) {
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

    PyObject *result = TransientDoubleVector_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}
