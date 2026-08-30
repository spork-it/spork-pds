#include "pds_internal.h"

// === VectorNode ===
PyTypeObject VectorNodeType;

static int VectorNode_traverse(VectorNode *self, visitproc visit, void *arg) {
    for (int i = 0; i < WIDTH; i++) {
        Py_VISIT(self->array[i]);
    }
    Py_VISIT(self->transient_id);
    return 0;
}

static int VectorNode_clear(VectorNode *self) {
    for (int i = 0; i < WIDTH; i++) {
        Py_CLEAR(self->array[i]);
    }
    Py_CLEAR(self->transient_id);
    return 0;
}

static void VectorNode_dealloc(VectorNode *self) {
    PyObject_GC_UnTrack(self);
    VectorNode_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

VectorNode *VectorNode_create(PyObject *transient_id) {
    VectorNode *node = (VectorNode *)VectorNodeType.tp_alloc(
        &VectorNodeType, 0);
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

PyTypeObject VectorNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.VectorNode",
    .tp_basicsize = sizeof(VectorNode),
    .tp_dealloc = (destructor)VectorNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)VectorNode_traverse,
    .tp_clear = (inquiry)VectorNode_clear,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

// Global empty node
VectorNode *EMPTY_NODE = NULL;

// === Vector ===
PyTypeObject VectorType;
Vector *EMPTY_VECTOR = NULL;

static int Vector_traverse(Vector *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->tail);
    Py_VISIT(self->transient_id);
    return 0;
}

static int Vector_clear(Vector *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->tail);
    Py_CLEAR(self->transient_id);
    return 0;
}

static void Vector_dealloc(Vector *self) {
    PyObject_GC_UnTrack(self);
    Vector_clear(self);
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
        if (self->tail == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->hash = 0;
        self->hash_computed = 0;
        self->transient_id = NULL;
        self->initialized = 0;
    }
    return (PyObject *)self;
}

Vector *Vector_create(Py_ssize_t cnt, int shift, VectorNode *root,
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

Py_ssize_t Vector_length(Vector *self) {
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
PyObject *Vector_nth_impl(Vector *self, Py_ssize_t i, PyObject *default_val) {
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
PyTypeObject TransientVectorType;
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
        PyObject *arr = Vector_array_for(self, i);
        if (!arr) return -1;

        PyObject *item = PyTuple_GET_ITEM(arr, i & MASK);
        Py_hash_t item_hash = PyObject_Hash(item);
        Py_DECREF(arr);

        if (item_hash == -1) return -1;
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
            new_cons->initialized = 1;

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

PyTypeObject VectorIteratorType;

static int VectorIterator_traverse(VectorIterator *self, visitproc visit, void *arg) {
    Py_VISIT(self->vec);
    Py_VISIT(self->cached_node);
    return 0;
}

static int VectorIterator_clear(VectorIterator *self) {
    Py_CLEAR(self->vec);
    Py_CLEAR(self->cached_node);
    return 0;
}

static void VectorIterator_dealloc(VectorIterator *self) {
    PyObject_GC_UnTrack(self);
    VectorIterator_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *VectorIterator_next(VectorIterator *self) {
    if (self->vec == NULL || self->index >= self->vec->cnt) {
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

PyTypeObject VectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.VectorIterator",
    .tp_basicsize = sizeof(VectorIterator),
    .tp_dealloc = (destructor)VectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)VectorIterator_traverse,
    .tp_clear = (inquiry)VectorIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)VectorIterator_next,
};

static PyObject *Vector_iter(Vector *self) {
    VectorIterator *it = (VectorIterator *)VectorIteratorType.tp_alloc(
        &VectorIteratorType, 0);
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

PyTypeObject VectorType = {
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
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)Vector_traverse,
    .tp_clear = (inquiry)Vector_clear,
    .tp_richcompare = (richcmpfunc)Vector_richcompare,
    .tp_iter = (getiterfunc)Vector_iter,
    .tp_methods = Vector_methods,
    .tp_init = (initproc)Vector_init,
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = Vector_new,
    .tp_free = PyObject_GC_Del,
};

// === TransientVector ===
typedef struct TransientVector {
    PyObject_HEAD
    Py_ssize_t cnt;
    int shift;
    VectorNode *root;
    PyObject *tail;  // list
    PyObject *id;
    uint64_t owner_thread_id;
} TransientVector;

static int TransientVector_traverse(TransientVector *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->tail);
    Py_VISIT(self->id);
    return 0;
}

static int TransientVector_clear(TransientVector *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->tail);
    Py_CLEAR(self->id);
    return 0;
}

static void TransientVector_dealloc(TransientVector *self) {
    PyObject_GC_UnTrack(self);
    TransientVector_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Vector_transient(Vector *self, PyObject *Py_UNUSED(ignored)) {
    TransientVector *t = (TransientVector *)TransientVectorType.tp_alloc(
        &TransientVectorType, 0);
    if (!t) return NULL;

    t->owner_thread_id = pds_current_thread_state_id();
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
    if (pds_check_transient_owner(self->owner_thread_id) < 0) {
        return;
    }
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

PyTypeObject TransientVectorIteratorType;

static int TransientVectorIterator_traverse(
    TransientVectorIterator *self, visitproc visit, void *arg) {
    Py_VISIT(self->tvec);
    return 0;
}

static int TransientVectorIterator_clear(TransientVectorIterator *self) {
    Py_CLEAR(self->tvec);
    return 0;
}

static void TransientVectorIterator_dealloc(TransientVectorIterator *self) {
    PyObject_GC_UnTrack(self);
    TransientVectorIterator_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *TransientVectorIterator_next(TransientVectorIterator *self) {
    if (self->tvec == NULL) {
        return NULL;
    }
    TransientVector_ensure_editable(self->tvec);
    if (PyErr_Occurred()) return NULL;

    if (self->index >= self->tvec->cnt) {
        return NULL;  // StopIteration
    }

    PyObject *result = TransientVector_sq_item(self->tvec, self->index);
    self->index++;
    return result;
}

PyTypeObject TransientVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientVectorIterator",
    .tp_basicsize = sizeof(TransientVectorIterator),
    .tp_dealloc = (destructor)TransientVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)TransientVectorIterator_traverse,
    .tp_clear = (inquiry)TransientVectorIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)TransientVectorIterator_next,
};

static PyObject *TransientVector_iter(TransientVector *self) {
    TransientVector_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    TransientVectorIterator *it =
        (TransientVectorIterator *)TransientVectorIteratorType.tp_alloc(
            &TransientVectorIteratorType, 0);
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

PyTypeObject TransientVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientVector",
    .tp_doc = "Transient vector for batch operations",
    .tp_basicsize = sizeof(TransientVector),
    .tp_dealloc = (destructor)TransientVector_dealloc,
    .tp_as_sequence = &TransientVector_as_sequence,
    .tp_as_mapping = &TransientVector_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)TransientVector_traverse,
    .tp_clear = (inquiry)TransientVector_clear,
    .tp_iter = (getiterfunc)TransientVector_iter,
    .tp_methods = TransientVector_methods,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

PyObject *pds_vec(PyObject *self, PyObject *args) {
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

                PyObject *result = TransientVector_persistent(t, NULL);
                Py_DECREF(t);
                return result;
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

    PyObject *result = TransientVector_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}
