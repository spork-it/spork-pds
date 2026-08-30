#include "pds_internal.h"

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

PyTypeObject SetType;
Set *EMPTY_SET = NULL;

static int Set_traverse(Set *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->transient_id);
    return 0;
}

static int Set_clear(Set *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->transient_id);
    return 0;
}

static void Set_dealloc(Set *self) {
    PyObject_GC_UnTrack(self);
    Set_clear(self);
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

Set *Set_create(Py_ssize_t cnt, PyObject *root, PyObject *transient_id) {
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

PyTypeObject SetIteratorType;

static int SetIterator_traverse(SetIterator *self, visitproc visit, void *arg) {
    Py_VISIT(self->kv_iter);
    return 0;
}

static int SetIterator_clear(SetIterator *self) {
    Py_CLEAR(self->kv_iter);
    return 0;
}

static void SetIterator_dealloc(SetIterator *self) {
    PyObject_GC_UnTrack(self);
    SetIterator_clear(self);
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

PyTypeObject SetIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.SetIterator",
    .tp_basicsize = sizeof(SetIterator),
    .tp_dealloc = (destructor)SetIterator_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)SetIterator_traverse,
    .tp_clear = (inquiry)SetIterator_clear,
    .tp_iter = PyObject_SelfIter,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
    .tp_iternext = (iternextfunc)SetIterator_next,
};

static PyObject *Set_iter(Set *self) {
    if (self->root == NULL) {
        return pds_empty_iterator();
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

    SetIterator *it = (SetIterator *)SetIteratorType.tp_alloc(
        &SetIteratorType, 0);
    if (!it) {
        Py_DECREF(kv_iter);
        return NULL;
    }

    it->kv_iter = kv_iter;
    return (PyObject *)it;
}

// Set operations: union, intersection, difference

// Forward declarations for TransientSet functions (defined later)
PyTypeObject TransientSetType;
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

    PDS_BEGIN_CRITICAL_SECTION(self);
    if (!self->hash_computed) {
        self->hash = h;
        self->hash_computed = 1;
    }
    cached_hash = self->hash;
    PDS_END_CRITICAL_SECTION();
    return cached_hash;
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
PyTypeObject TransientSetType;

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
        new_cons->initialized = 1;

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

PyTypeObject SetType = {
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
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)Set_traverse,
    .tp_clear = (inquiry)Set_clear,
    .tp_richcompare = (richcmpfunc)Set_richcompare,
    .tp_iter = (getiterfunc)Set_iter,
    .tp_methods = Set_methods,
    .tp_init = (initproc)Set_init,
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = Set_new,
    .tp_free = PyObject_GC_Del,
};

// === TransientSet ===
struct TransientSet {
    PyObject_HEAD
    Py_ssize_t cnt;
    PyObject *root;
    PyObject *id;
};

static int TransientSet_traverse(
    TransientSet *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->id);
    return 0;
}

static int TransientSet_clear_gc(TransientSet *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->id);
    return 0;
}

static void TransientSet_dealloc(TransientSet *self) {
    PyObject_GC_UnTrack(self);
    TransientSet_clear_gc(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Set_transient(Set *self, PyObject *Py_UNUSED(ignored)) {
    TransientSet *t = (TransientSet *)TransientSetType.tp_alloc(
        &TransientSetType, 0);
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
        return pds_empty_iterator();
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

PyTypeObject TransientSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientSet",
    .tp_doc = "Transient set for batch operations",
    .tp_basicsize = sizeof(TransientSet),
    .tp_dealloc = (destructor)TransientSet_dealloc,
    .tp_as_sequence = &TransientSet_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)TransientSet_traverse,
    .tp_clear = (inquiry)TransientSet_clear_gc,
    .tp_iter = (getiterfunc)TransientSet_iter,
    .tp_methods = TransientSet_methods,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

PyObject *pds_set(PyObject *self, PyObject *args) {
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
    TransientSet *t = (TransientSet *)TransientSetType.tp_alloc(
        &TransientSetType, 0);
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
