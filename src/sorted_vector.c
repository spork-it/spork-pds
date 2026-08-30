#include "pds_internal.h"

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

PyTypeObject RBNodeType;

static int RBNode_traverse(RBNode *self, visitproc visit, void *arg) {
    Py_VISIT(self->value);
    Py_VISIT(self->sort_key);
    Py_VISIT(self->left);
    Py_VISIT(self->right);
    Py_VISIT(self->edit);
    return 0;
}

static int RBNode_clear(RBNode *self) {
    Py_CLEAR(self->value);
    Py_CLEAR(self->sort_key);
    Py_CLEAR(self->left);
    Py_CLEAR(self->right);
    Py_CLEAR(self->edit);
    return 0;
}

static void RBNode_dealloc(RBNode *self) {
    PyObject_GC_UnTrack(self);
    RBNode_clear(self);
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
    RBNode *node = (RBNode *)RBNodeType.tp_alloc(&RBNodeType, 0);
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

    RBNode *new_node = (RBNode *)RBNodeType.tp_alloc(&RBNodeType, 0);
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

PyTypeObject RBNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.RBNode",
    .tp_doc = "Red-Black Tree Node (internal)",
    .tp_basicsize = sizeof(RBNode),
    .tp_dealloc = (destructor)RBNode_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)RBNode_traverse,
    .tp_clear = (inquiry)RBNode_clear,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
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

PyTypeObject SortedVectorType;

static int SortedVector_traverse(
    SortedVector *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->key_fn);
    return 0;
}

static int SortedVector_clear(SortedVector *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->key_fn);
    return 0;
}

static void SortedVector_dealloc(SortedVector *self) {
    PyObject_GC_UnTrack(self);
    SortedVector_clear(self);
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
    SortedVector *result = (SortedVector *)SortedVectorType.tp_alloc(
        &SortedVectorType, 0);
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

PyTypeObject SortedVectorIteratorType;

static int SortedVectorIterator_traverse(
    SortedVectorIterator *self, visitproc visit, void *arg) {
    for (Py_ssize_t i = 0; i < self->stack_size; i++) {
        Py_VISIT(self->stack[i]);
    }
    return 0;
}

static int SortedVectorIterator_clear(SortedVectorIterator *self) {
    for (Py_ssize_t i = 0; i < self->stack_size; i++) {
        Py_CLEAR(self->stack[i]);
    }
    PyMem_Free(self->stack);
    self->stack = NULL;
    self->stack_size = 0;
    self->stack_capacity = 0;
    return 0;
}

static void SortedVectorIterator_dealloc(SortedVectorIterator *self) {
    PyObject_GC_UnTrack(self);
    SortedVectorIterator_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int SortedVectorIterator_push_left(
    SortedVectorIterator *self, RBNode *node) {
    while (node) {
        if (self->stack_size >= self->stack_capacity) {
            Py_ssize_t new_cap = self->stack_capacity * 2;
            RBNode **new_stack = PyMem_Realloc(
                self->stack, new_cap * sizeof(RBNode *));
            if (!new_stack) {
                PyErr_NoMemory();
                return -1;
            }
            self->stack = new_stack;
            self->stack_capacity = new_cap;
        }
        Py_INCREF(node);
        self->stack[self->stack_size++] = node;
        node = node->left;
    }
    return 0;
}

static PyObject *SortedVectorIterator_next(SortedVectorIterator *self) {
    if (self->stack_size == 0) {
        return NULL;  // StopIteration
    }

    RBNode *node = self->stack[--self->stack_size];
    PyObject *value = node->value;
    Py_INCREF(value);

    // Push left spine of right subtree
    if (SortedVectorIterator_push_left(self, node->right) < 0) {
        Py_DECREF(value);
        Py_DECREF(node);
        return NULL;
    }

    Py_DECREF(node);
    return value;
}

PyTypeObject SortedVectorIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.SortedVectorIterator",
    .tp_basicsize = sizeof(SortedVectorIterator),
    .tp_dealloc = (destructor)SortedVectorIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)SortedVectorIterator_traverse,
    .tp_clear = (inquiry)SortedVectorIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)SortedVectorIterator_next,
};

static PyObject *SortedVector_iter(SortedVector *self) {
    SortedVectorIterator *iter =
        (SortedVectorIterator *)SortedVectorIteratorType.tp_alloc(
            &SortedVectorIteratorType, 0);
    if (!iter) return NULL;

    // Initial capacity based on expected tree depth
    iter->stack_capacity = 32;
    iter->stack = PyMem_Malloc(iter->stack_capacity * sizeof(RBNode *));
    if (!iter->stack) {
        Py_DECREF(iter);
        return PyErr_NoMemory();
    }
    iter->stack_size = 0;

    if (SortedVectorIterator_push_left(iter, self->root) < 0) {
        Py_DECREF(iter);
        return NULL;
    }

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
    Py_uhash_t hash = 0x345678;
    Py_uhash_t mult = 1000003;

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
        hash = (hash ^ (Py_uhash_t)item_hash) * mult;
        mult += (Py_uhash_t)82520 + (Py_uhash_t)2 * (Py_uhash_t)self->cnt;
    }
    Py_DECREF(iter);

    hash += (Py_uhash_t)97531;
    return pds_finalize_hash(hash);
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

    SortedVector *result = (SortedVector *)SortedVectorType.tp_alloc(
        &SortedVectorType, 0);
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
    uint64_t owner_thread_id;
} TransientSortedVector;

PyTypeObject TransientSortedVectorType;

static int TransientSortedVector_traverse(
    TransientSortedVector *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->key_fn);
    Py_VISIT(self->id);
    return 0;
}

static int TransientSortedVector_clear_gc(TransientSortedVector *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->key_fn);
    Py_CLEAR(self->id);
    return 0;
}

static void TransientSortedVector_dealloc(TransientSortedVector *self) {
    PyObject_GC_UnTrack(self);
    TransientSortedVector_clear_gc(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static void TransientSortedVector_ensure_editable(TransientSortedVector *self) {
    if (pds_check_transient_owner(self->owner_thread_id) < 0) {
        return;
    }
    if (!self->id) {
        PyErr_SetString(PyExc_RuntimeError, "TransientSortedVector already made persistent");
    }
}

static PyObject *SortedVector_transient(SortedVector *self, PyObject *Py_UNUSED(ignored)) {
    TransientSortedVector *t =
        (TransientSortedVector *)TransientSortedVectorType.tp_alloc(
            &TransientSortedVectorType, 0);
    if (!t) return NULL;

    t->owner_thread_id = pds_current_thread_state_id();
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

    SortedVector *result = (SortedVector *)SortedVectorType.tp_alloc(
        &SortedVectorType, 0);
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

PyTypeObject TransientSortedVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientSortedVector",
    .tp_doc = "Transient sorted vector for batch operations",
    .tp_basicsize = sizeof(TransientSortedVector),
    .tp_dealloc = (destructor)TransientSortedVector_dealloc,
    .tp_as_sequence = &TransientSortedVector_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)TransientSortedVector_traverse,
    .tp_clear = (inquiry)TransientSortedVector_clear_gc,
    .tp_methods = TransientSortedVector_methods,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
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

PyTypeObject SortedVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.SortedVector",
    .tp_doc = "Persistent sorted vector (Red-Black Tree)",
    .tp_basicsize = sizeof(SortedVector),
    .tp_dealloc = (destructor)SortedVector_dealloc,
    .tp_repr = (reprfunc)SortedVector_repr,
    .tp_as_sequence = &SortedVector_as_sequence,
    .tp_as_mapping = &SortedVector_as_mapping,
    .tp_hash = (hashfunc)SortedVector_hash,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)SortedVector_traverse,
    .tp_clear = (inquiry)SortedVector_clear,
    .tp_iter = (getiterfunc)SortedVector_iter,
    .tp_richcompare = (richcmpfunc)SortedVector_richcompare,
    .tp_methods = SortedVector_methods,
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = SortedVector_new,
    .tp_init = (initproc)SortedVector_init,
    .tp_free = PyObject_GC_Del,
};

// Empty sorted vector constant
SortedVector *EMPTY_SORTED_VECTOR = NULL;

PyObject *pds_sorted_vec(PyObject *self, PyObject *args, PyObject *kwargs) {
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
