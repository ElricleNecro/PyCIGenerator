import argparse as ag
import Generate as g

description = """
Génère les conditions initiales selon certains paramètres.
"""

def main(opt):
	pass

if __name__ == '__main__':
	parser = ag.ArgumentParser(description=description)

	opt = parser.parse_args()

	main(opt)
