#include "pds_internal.h"

// === Cons ===
PyTypeObject ConsType;

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

PyTypeObject ConsIteratorType;

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

PyTypeObject ConsIteratorType = {
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

PyTypeObject ConsType = {
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

PyObject *pds_cons(PyObject *self, PyObject *args) {
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
