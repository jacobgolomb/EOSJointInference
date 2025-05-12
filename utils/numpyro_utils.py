import numpyro
import numpyro.distributions as dist

def decode_string_into_args_kwargs(expression):
    argumentslist = expression.split(',')
    argumentslist = [arg.strip() for arg in argumentslist]

    args = []
    kwargstr = ""
    for arg in argumentslist:
        if "=" in arg:
            kwargstr += arg + ","
        else:
            if arg.startswith("np."):
                arg_float = eval(arg)
            else:
                arg_float = float(arg)
            args.append(arg_float)
    kwargs = eval(f"dict({kwargstr})")

    return args, kwargs

def get_numpyro_dist_from_string(expression):
    funcstring, inputstring = expression.split("(", 1)
    distfunc = getattr(dist, funcstring.strip())
    argstring = inputstring.strip()[:-1]
    args, kwargs = decode_string_into_args_kwargs(argstring)

    return distfunc(*args, **kwargs)

def parse_input_line(line):
    variable, functionstr = line.split("=", 1)
    variable = variable.strip()

    try:
        if functionstr.startswith("np."):
            float_arg = eval(functionstr)
        else:
            float_arg = float(functionstr)
        return variable, float_arg

    except ValueError:
        pass
    function = get_numpyro_dist_from_string(functionstr)
    return variable, function

def get_priors_from_file(filename):
    prior = dict()
    with open(filename, "r") as priorfile:
        lines = priorfile.read().splitlines()
    for line in lines:
        param, func = parse_input_line(line)
        prior[param] = func
    return prior

def sample_parameters_from_dict(prior):
    samples = dict()
    for param in prior:
        if isinstance(prior[param], float):
            samples[param] = numpyro.deterministic(param, prior[param])
        else:
            samples[param] = numpyro.sample(param, prior[param])
    return samples