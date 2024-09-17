from click.testing import CliRunner
from hepconvert import __main__

def test_hello_world():
  runner = CliRunner()
  result = runner.invoke(__main__.copy_root, [args, command])
  assert result.exit_code == 0 # ?
  assert result.output == 'Hello Peter!\n'
