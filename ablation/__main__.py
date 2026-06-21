import tyro

from ablation.studies import (
    candidate_size,
    frac_hard,
    gsk_hepg2_data_fraction,
)

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "candidate-size": candidate_size,
            "frac-hard": frac_hard,
            "gsk-hepg2-data-fraction": gsk_hepg2_data_fraction,
        }
    )
