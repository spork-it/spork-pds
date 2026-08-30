#include "pds_internal.h"

/* A minimal type for identity/sentinel objects. */
typedef struct {
    PyObject_HEAD
} PdsSentinel;

static void
PdsSentinel_dealloc(PdsSentinel *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyTypeObject PdsSentinelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "spork_pds._Sentinel",
    .tp_basicsize = sizeof(PdsSentinel),
    .tp_dealloc = (destructor)PdsSentinel_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
};

PyObject *_MISSING = NULL;

/* Generic __class_getitem__ for type annotations. */
PyObject *
Generic_class_getitem(PyObject *cls, PyObject *args)
{
    PyObject *typing_module = PyImport_ImportModule("typing");
    if (typing_module == NULL) {
        return NULL;
    }

    PyObject *generic_alias_func =
        PyObject_GetAttrString(typing_module, "_GenericAlias");
    Py_DECREF(typing_module);

    if (generic_alias_func == NULL) {
        PyErr_Clear();
        return Py_BuildValue("(OO)", cls, args);
    }

    PyObject *args_tuple;
    if (PyTuple_Check(args)) {
        args_tuple = args;
        Py_INCREF(args_tuple);
    }
    else {
        args_tuple = PyTuple_Pack(1, args);
        if (args_tuple == NULL) {
            Py_DECREF(generic_alias_func);
            return NULL;
        }
    }

    PyObject *result = PyObject_CallFunctionObjArgs(
        generic_alias_func, cls, args_tuple, NULL);
    Py_DECREF(generic_alias_func);
    Py_DECREF(args_tuple);
    return result;
}

PyObject *
pds_empty_iterator(void)
{
    PyObject *empty = PyTuple_New(0);
    if (empty == NULL) {
        return NULL;
    }
    PyObject *iterator = PyObject_GetIter(empty);
    Py_DECREF(empty);
    return iterator;
}

int
ctpop(unsigned int i)
{
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

int
mask_hash(Py_hash_t hash_val, int shift)
{
    return (hash_val >> shift) & MASK;
}

unsigned int
bitpos(Py_hash_t hash_val, int shift)
{
    return 1U << mask_hash(hash_val, shift);
}

int
bitmap_index(unsigned int bitmap, unsigned int bit)
{
    return ctpop(bitmap & (bit - 1));
}
