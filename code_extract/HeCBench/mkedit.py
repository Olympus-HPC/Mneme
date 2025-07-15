import re
from pathlib import Path

class Target:
    def __init__(self, name, deps = [], cmds = []):
        self.name = name
        self.deps = deps
        self.cmds = cmds

    def __repr__(self) -> str:
        return f"{self.name}: {' '.join(self.deps)} > {'   '.join(self.cmds)}"

class MakefileEditor:
    def __init__(self, filename):
        self.filename = filename
        self.variables = {}
        self.new_variables = {}
        self.targets = {}
        self.modified_targets = set()
        self.original_lines = []  # Store the original lines to detect modifications
        self.var_regex = r'^([A-Za-z0-9_]+)\s*(\?=|:=|::=|=)\s*(.*?)\s*$'
        self.target_regex = r'^([^#\s]*?)\s*:\s*(.*?)\s*$'
        self.command_regex = r'^\t([^\s].*)\s*$'
        self.include_regex = r'^include\s+([^\s]*?)\s*$'
        self.load_makefile()

    def load_makefile(self):
        """Load the Makefile into lines and extract variables and targets with commands."""

        with open(self.filename, 'r') as f:
            self.original_lines = f.readlines()

        current_target = None

        # Parse the lines for variables, targets, and their commands
        current_var = None
        current_deps_target = None
        current_cmd_target = None
        i = 0
        while i < len(self.original_lines):
            line = self.original_lines[i]
            curr_index = i
            i += 1

            if current_var is not None:
                value = line.strip()
                var_name = current_var
                if value.endswith('\\'):
                    value = value.rstrip('\\').strip()
                else:
                    current_var = None
                ends_with_space = self.variables[var_name].endswith(' ')
                self.variables[var_name] += ("" if ends_with_space else "  ") + value
                continue
            elif current_deps_target is not None:
                value = line.strip()
                target_name = current_deps_target
                if value.endswith('\\'):
                    value = value.rstrip('\\')
                else:
                    current_deps_target = None
                self.targets[target_name][-1].deps.append(value)
                continue
            elif current_cmd_target is not None:
                value = line.strip()
                target_name = current_cmd_target
                if value.endswith('\\'):
                    value = value.rstrip('\\')
                else:
                    current_cmd_target = None
                self.targets[target_name][-1].cmds[-1] += " " + value
                continue

            
            # Check for includes
            match_inc = re.match(self.include_regex, line)
            if match_inc:
                self.original_lines[curr_index] = "#" + line
                fname = match_inc.group(1)
                with open(Path(self.filename).parent / fname, 'r') as f:
                    lines = f.readlines()
                self.original_lines[i:i] = lines


            # Check for variable assignments
            match_var = re.match(self.var_regex, line)
            if match_var:
                current_target = None
                var_name = match_var.group(1)
                var_value = match_var.group(3).strip()
                if var_value.endswith('\\'):
                    var_value = var_value.rstrip('\\').strip()
                    current_var = var_name
                self.variables[var_name] = var_value
                continue

            # Check for target definitions
            match_target = re.match(self.target_regex, line)
            if match_target:
                target_name = match_target.group(1)
                deps =  match_target.group(2)
                if deps.endswith("\\"):
                    deps = deps.rstrip("\\")
                    current_deps_target = target_name
                dependencies = deps.strip().split()
                if target_name not in self.targets:
                    self.targets[target_name] = []
                self.targets[target_name].append(Target(target_name, dependencies, cmds=[]))
                current_target = target_name
                continue

            # If we are in a target block, capture its commands
            if current_target:
                match_command = re.match(self.command_regex, line)
                if match_command:
                    cmd = match_command.group(1).strip()
                    if cmd.endswith("\\"):
                        cmd = cmd.rstrip("\\").strip()
                        current_cmd_target = current_target
                    self.targets[current_target][-1].cmds.append(cmd)
                else:
                    # No more commands for this target, move to next line
                    current_target = None

    def output_makefile(self, out_file_name=None):
        """Print/Save the Makefile including modified and newly added variables and targets."""
        lines = []
        # Append new variables
        for var_name, value in self.new_variables.items():
            lines.append(f"{var_name} = {value}\n")

        current_target = None
        line_has_val = False
        target_index = {}

        for line in self.original_lines:
            if line_has_val:
                if not line.strip().endswith('\\'):
                    line_has_val = False
                continue

            match_var = re.match(self.var_regex, line)
            if match_var:
                current_target = None
                var_name = match_var.group(1)
                if var_name in self.variables:
                    original_value = match_var.group(3).strip()
                    if original_value.endswith('\\'):
                        line_has_val = True

                    if original_value != self.variables[var_name]:
                        line = f"{var_name} = {self.variables[var_name]}\n"
                lines.append(line)
                continue

            match_target = re.match(self.target_regex, line)
            if match_target:
                current_target = match_target.group(1)
                if match_target.group(2).strip().endswith('\\'):
                    line_has_val = True

                if current_target not in target_index:
                    target_index[current_target] = 0
                else:
                    target_index[current_target] += 1

                updated_target = self.targets[current_target][target_index[current_target]]
                dependencies = " ".join(updated_target.deps)
                lines.append(f"{current_target}: {dependencies}\n")
                for command in updated_target.cmds:
                    lines.append(f"\t{command}\n")

                continue

            if current_target:
                if line.strip():
                    if not line.startswith('\t'):
                        current_target = None
                    else:
                        if line.strip().endswith('\\'):
                            line_has_val = True
                        continue  # Skip original commands; the modified ones are already written

            lines.append(line)

        if out_file_name is None:
            for line in lines:
                print(line, end="")
            return

        with open(out_file_name, 'w') as f:
            f.writelines(lines)

    def set_variable(self, var_name, value, add_if_new=True):
        """Modify the value of a variable."""
        if var_name in self.variables:
            self.variables[var_name] = value
        elif add_if_new:
            self.new_variables[var_name] = value

    def get_variable(self, var_name):
        """Return the value of a variable."""
        if var_name in self.variables:
            return self.variables[var_name]
        else:
            return None

    def get_target(self, target_name):
        """Return the dependcencies and commands of a target."""
        if target_name in self.targets:
            return self.targets[target_name]
        else:
            return None
 
    def get_variables(self):
        """Return the dictionary of variables."""
        return self.variables

    def get_targets(self):
        """Return the dictionary of targets."""
        return self.targets


# Example usage
if __name__ == '__main__':
    import sys

    argv = sys.argv
    if len(argv) < 2:
        makefile_path = 'Makefile'
    else:
        makefile_path = argv[1]

    # Create an instance of MakefileEditor
    editor = MakefileEditor(makefile_path)

    # Print the current variables and targets
    print("Current Variables:", editor.get_variables())
    print("Current Targets:", editor.get_targets())

    # Modify variables
    editor.set_variable('CC', 'clang')
    # editor.set_variable('CFLAGS', '-O2 -Wall')

    # Modify targets (adding/removing dependencies and commands)
    # Add a new variable
    editor.set_variable('LIBRARY2', '-lm')

    clean = editor.get_target('clean')
    if clean is not None:
        clean[0].deps = ["hi"]
        clean[0].cmds = ["rm -rf *"]

    # Add a new target with commands
    print("Modified Variables:", editor.get_variables())
    print("Modified Targets:", editor.get_targets())
    # Save the updated Makefile (only modified variables will be written)
    editor.output_makefile()


