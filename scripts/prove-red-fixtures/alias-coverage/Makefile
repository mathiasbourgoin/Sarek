# Fixture invoker for scripts/check-test-alias-coverage.sh's DIRECTION 3.
#
# DIRECTION 3 asks whether each declared alias is REACHABLE from something that
# actually builds it. With no invoker file under the scanned root at all, the
# gate exits 2 rather than reporting OK -- refusing a vacuous pass. That refusal
# is correct, and it is what broke this spec's baseline when DIRECTION 3 landed:
# the fixture tree had no invoker, so the unmutated subject exited 2.
#
# This target is the invoker. The `e2e-fixture` alias is not self-executing, so
# without this line it is unreachable and DIRECTION 3 must say so -- which is
# exactly what the `alias-unreachable` mutation checks.
fixture-e2e:
	dune build @e2e-fixture
