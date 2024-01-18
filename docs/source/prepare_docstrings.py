from __future__ import annotations

import importlib
import inspect
import os.path
import pkgutil
import subprocess
import sys

import odapt

order = [
    "odapt",
    "odapt.histogram_adding",
    "odapt.merge",
    "odapt.parquet_to_root",
    "odapt.root_to_parquet",
    "odapt.copy_root",
]

common = [
    "odapt.parquet_to_root.parquet_to_root",
    "odapt.root_to_parquet.root_to_parquet",
    "odapt.copy_root.copy_root",
    "odapt.merge.hadd_and_merge",
    "odapt.histogram_adding.hadd",
]

latest_commit = (
    subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)
main = open("main.toctree", "w")
main.write(
    """.. toctree::
    :caption: Main Interface
    :hidden:

{}""".format(
        "".join(f"    {x}\n" for x in common)
    )
)
toctree = open("odapt.toctree", "w")
toctree.write(
    """.. toctree::
    :caption: Detailed Reference
    :hidden:

"""
)
toctree2 = None


def ensure(filename, content):
    overwrite = not os.path.exists(filename)
    if not overwrite:
        overwrite = open(filename).read() != content
    if overwrite:
        open(filename, "w").write(content)
        sys.stderr.write(filename + " (OVERWRITTEN)\n")
    else:
        sys.stderr.write(filename + "\n")


def handle_module(modulename, module):
    if any(x.startswith("test") and x.startswith("_") for x in modulename.split(".")):
        return

    content = """{0}
{1}

.. automodule:: {0}
""".format(
        modulename, "=" * len(modulename)
    )
    ensure(modulename + ".rst", content)
    if toctree2 is None:
        toctree.write("    " + modulename + " (module) <" + modulename + ">\n")
    else:
        toctree2.write("    " + modulename + " (module) <" + modulename + ">\n")

    if modulename != "odapt" and all(
        not x.startswith("test") and not x.startswith("_")
        for x in modulename.split(".")
    ):

        def good(obj):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if obj.__module__ == modulename:
                    return True
            return False

        def line_order(pair):
            name, obj = pair
            try:
                return inspect.getsourcelines(obj)[1]
            except OSError:
                return float("inf")

        for pair in sorted(inspect.getmembers(module, good), key=line_order):
            name, obj = pair
            if not name.startswith("_") and not name.startswith("test"):
                if inspect.isclass(obj):
                    handle_class(modulename + "." + name, obj)
                elif inspect.isfunction(obj):
                    handle_function(modulename + "." + name, obj)


def handle_class(classname, cls):
    def line_order(obj):
        if isinstance(obj, property):
            obj = obj.fget
        return inspect.getsourcelines(obj)[1]

    methods = {}
    mro = list(cls.__mro__)

    if hasattr(odapt, cls.__name__):
        title = "odapt." + cls.__name__
        upfront = True
    else:
        title = classname
        upfront = False

    for index, basecls in enumerate(mro):
        if basecls.__module__.startswith("odapt."):

            def good(obj):
                if inspect.ismethod(obj) or inspect.isfunction(obj):
                    module, name = obj.__module__, obj.__name__
                elif isinstance(obj, property):
                    try:
                        module, name = obj.fget.__module__, obj.fget.__name__
                    except AttributeError:
                        return False
                else:
                    module, name = "", ""
                if module.startswith("odapt."):
                    if index + 1 >= len(mro) or obj is not getattr(
                        mro[index + 1], name, None
                    ):
                        return True
                return False

            for name, obj in inspect.getmembers(basecls, good):
                if (
                    name in basecls.__dict__
                    and not name.startswith("_")
                    and not name.startswith("test")
                ):
                    fill = []
                    fill.append(".. _" + classname + "." + name + ":" + "\n")
                    fill.append(name)
                    fill.append("-" * len(fill[-1]))
                    fill.append("")
                    if basecls is not cls:
                        fill.append(
                            "Inherited from :doc:`{}`.".format(
                                basecls.__module__ + "." + basecls.__name__
                            )
                        )
                        fill.append("")
                    if isinstance(obj, property):
                        fill.append(".. autoattribute:: " + classname + "." + name)
                    elif callable(obj):
                        fill.append(".. automethod:: " + classname + "." + name)
                    else:
                        fill.append(".. autoattribute:: " + classname + "." + name)
                    fill.append("")
                    methods[name] = (index, line_order(obj), "\n".join(fill))

    def prettymro(c):
        fullname = c.__module__ + "." + c.__name__
        if c.__module__.startswith("odapt."):
            return "#. :doc:`" + fullname + "`"
        else:
            return "#. ``" + fullname + "``"

    fullfilename = importlib.import_module(cls.__module__).__file__
    shortfilename = fullfilename[fullfilename.rindex("odapt/") :]
    link = "`{} <https://github.com/zbilodea/odapt/blob/{}/src/{}>`__".format(
        cls.__module__, latest_commit, shortfilename
    )
    try:
        linelink = "`line {0} <https://github.com/zbilodea/odapt/blob/{1}/src/{2}#L{0}>`__".format(
            inspect.getsourcelines(cls)[1], latest_commit, shortfilename
        )
    except OSError:
        linelink = ""

    inheritance_header = ""
    inheritance_footer = ""
    inheritance_sep = ""
    inheritance = [prettymro(c) for c in cls.__mro__[1:] if c is not object]
    if len(inheritance) > 0:
        longest_cell = max(*(len(x) for x in inheritance), 22)
        inheritance_header = """.. table::
    :class: note

    +-{}-+
    | **Inheritance order:** {}|
    +={}=+
    | """.format(
            "-" * longest_cell, " " * (longest_cell - 22), "=" * longest_cell
        )
        inheritance_footer = """ |
    +-{}-+""".format(
            "-" * longest_cell
        )
        inheritance = [x + " " * (longest_cell - len(x)) for x in inheritance]
        inheritance_sep = """ |
    | """

    content = """{}
{}

Defined in {} on {}.

{}

.. autoclass:: {}

{}
""".format(
        title,
        "=" * len(title),
        link,
        linelink,
        inheritance_header + inheritance_sep.join(inheritance) + inheritance_footer,
        classname,
        "\n".join([text for index, line, text in sorted(methods.values())]),
    )

    ensure(classname + ".rst", content)
    if upfront or toctree2 is None:
        if classname not in common:
            toctree.write("    " + classname + "\n")
        toctree2.write("    " + classname + " <" + classname + ">\n")
    else:
        toctree2.write("    " + classname + "\n")


def handle_function(functionname, cls):
    if hasattr(odapt, cls.__name__):
        title = "odapt." + cls.__name__
        upfront = True
    else:
        title = functionname
        upfront = False

    fullfilename = importlib.import_module(cls.__module__).__file__
    shortfilename = fullfilename[fullfilename.rindex("odapt/") :]
    link = "`{} <https://github.com/zbilodea/odapt/blob/{}/src/{}>`__".format(
        cls.__module__, latest_commit, shortfilename
    )
    linelink = (
        "`line {0} <https://github.com/zbilodea/odapt/blob/{1}/src/{2}#L{0}>`__".format(
            inspect.getsourcelines(cls)[1], latest_commit, shortfilename
        )
    )

    content = """{}
{}

Defined in {} on {}.

.. autofunction:: {}
""".format(
        title, "=" * len(title), link, linelink, functionname
    )
    ensure(functionname + ".rst", content)
    if upfront or toctree2 is None:
        if functionname not in common:
            toctree.write("    " + functionname + "\n")
        toctree2.write("    " + functionname + " <" + functionname + ">\n")
    else:
        toctree2.write("    " + functionname + "\n")


for modulename in order:
    module = importlib.import_module(modulename)

    if modulename != "odapt":
        toctree2 = open(modulename + ".toctree", "w")
        toctree2.write(
            """.. toctree::
    :caption: {}
    :hidden:

""".format(
                modulename.replace("odapt.", "")
            )
        )

    handle_module(modulename, module)
    if module.__file__.endswith("__init__.py") and modulename != "odapt":
        for submodulename in sorted(
            modulename + "." + name
            for loader, name, is_pkg in pkgutil.walk_packages(module.__path__)
        ):
            submodule = importlib.import_module(submodulename)
            handle_module(submodulename, submodule)

toctree.close()
toctree2.close()