import tyro

from ablation.studies import (
    db_malaria_candidate_size,
    db_malaria_frac_hard,
    gsk_hepg2_data_fraction,
)

if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "db-malaria-candidate-size": db_malaria_candidate_size,
            "db-malaria-frac-hard": db_malaria_frac_hard,
            "gsk-hepg2-data-fraction": gsk_hepg2_data_fraction,
        }
    )
