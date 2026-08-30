#include "pds_internal.h"

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

PyTypeObject MapType;
Map *EMPTY_MAP = NULL;

static int Map_traverse(Map *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->transient_id);
    return 0;
}

static int Map_clear(Map *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->transient_id);
    return 0;
}

static void Map_dealloc(Map *self) {
    PyObject_GC_UnTrack(self);
    Map_clear(self);
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

Map *Map_create(Py_ssize_t cnt, PyObject *root, PyObject *transient_id) {
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
        return pds_empty_iterator();
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
        return pds_empty_iterator();
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
        return pds_empty_iterator();
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

        h += (Py_uhash_t)kh ^ (Py_uhash_t)vh;
    }
    Py_DECREF(items_iter);

    if (PyErr_Occurred()) return -1;

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
PyTypeObject TransientMapType;

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
        new_cons->initialized = 1;

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
PyObject *pds_hash_map(PyObject *self, PyObject *args);

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

PyTypeObject MapType = {
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
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)Map_traverse,
    .tp_clear = (inquiry)Map_clear,
    .tp_richcompare = (richcmpfunc)Map_richcompare,
    .tp_iter = (getiterfunc)Map_iter,
    .tp_methods = Map_methods,
    .tp_init = (initproc)Map_init,
    .tp_alloc = PyType_GenericAlloc,
    .tp_new = Map_new,
    .tp_free = PyObject_GC_Del,
};

// === TransientMap ===
typedef struct TransientMap {
    PyObject_HEAD
    Py_ssize_t cnt;
    PyObject *root;
    PyObject *id;
    uint64_t owner_thread_id;
} TransientMap;

static int TransientMap_traverse(
    TransientMap *self, visitproc visit, void *arg) {
    Py_VISIT(self->root);
    Py_VISIT(self->id);
    return 0;
}

static int TransientMap_clear(TransientMap *self) {
    Py_CLEAR(self->root);
    Py_CLEAR(self->id);
    return 0;
}

static void TransientMap_dealloc(TransientMap *self) {
    PyObject_GC_UnTrack(self);
    TransientMap_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Map_transient(Map *self, PyObject *Py_UNUSED(ignored)) {
    TransientMap *t = (TransientMap *)TransientMapType.tp_alloc(
        &TransientMapType, 0);
    if (!t) return NULL;

    t->owner_thread_id = pds_current_thread_state_id();
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
    if (pds_check_transient_owner(self->owner_thread_id) < 0) {
        return;
    }
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

static PyObject *TransientMap_keys(TransientMap *self, PyObject *Py_UNUSED(ignored)) {
    return TransientMap_iter(self);
}

static PyObject *TransientMap_values(TransientMap *self, PyObject *Py_UNUSED(ignored)) {
    TransientMap_ensure_editable(self);
    if (PyErr_Occurred()) return NULL;

    if (self->root == NULL) {
        return pds_empty_iterator();
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
        return pds_empty_iterator();
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

PyTypeObject TransientMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds.TransientMap",
    .tp_doc = "Transient map for batch operations",
    .tp_basicsize = sizeof(TransientMap),
    .tp_dealloc = (destructor)TransientMap_dealloc,
    .tp_as_sequence = &TransientMap_as_sequence,
    .tp_as_mapping = &TransientMap_as_mapping,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_traverse = (traverseproc)TransientMap_traverse,
    .tp_clear = (inquiry)TransientMap_clear,
    .tp_iter = (getiterfunc)TransientMap_iter,
    .tp_methods = TransientMap_methods,
    .tp_alloc = PyType_GenericAlloc,
    .tp_free = PyObject_GC_Del,
};

PyObject *pds_hash_map(PyObject *self, PyObject *args) {
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

    PyObject *result = TransientMap_persistent(t, NULL);
    Py_DECREF(t);
    return result;
}
