from deep_numerical.neural.collision_op.fno   import ConservativeFNO
from deep_numerical.neural.collision_op.pfno  import PFNO
from deep_numerical.neural.collision_op.prfno import PRFNO
from deep_numerical.neural.collision_op.rfno  import ConservativeRFNO


__all__: list[str] = [
    "PFNO",
    "PRFNO",
    "ConservativeFNO",
    "ConservativeRFNO",
]


##################################################
def main() -> None:
    pass


if __name__ == '__main__':
    main()
