# Makefile

# Default target
submit:

# Define the Python command to execute
PYTHON_CMD = python3 run_condor.py command="python3 your_program.py"

# Define the list of arguments to pass to the Python script
ARGS := $(foreach var, $(filter VAR_%, $(.VARIABLES)), --$(subst VAR_,,$(var))=$(value $(var)))

# Define the "run" target
submit:
    @echo $(PYTHON_CMD) $(ARGS)

