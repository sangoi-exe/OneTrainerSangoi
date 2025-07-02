from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

class AddVariationInfo(PipelineModule, RandomAccessPipelineModule):
    """
    Um módulo simples que adiciona o índice da variação atual
    e o nome do conceito ao item do batch.
    """
    def __init__(self, variation_out_name: str, concept_name_in_name: str, concept_name_out_name: str):
        super().__init__()
        self.variation_out_name = variation_out_name
        self.concept_name_in_name = concept_name_in_name
        self.concept_name_out_name = concept_name_out_name

    def length(self) -> int:
        return self._get_previous_length(self.concept_name_in_name)

    def get_inputs(self) -> list[str]:
        return [self.concept_name_in_name]

    def get_outputs(self) -> list[str]:
        return [self.variation_out_name, self.concept_name_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        # Pega o nome do conceito do módulo anterior
        concept_name = self._get_previous_item(variation, self.concept_name_in_name, index)
        
        # A informação crucial: `variation` é o índice da variação atual.
        # Nós simplesmente o retornamos.
        return {
            self.variation_out_name: variation,
            self.concept_name_out_name: concept_name
        }