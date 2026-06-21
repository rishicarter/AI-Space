def calculate_llm_price(input_tokens, output_tokens, model_name="Unknown", input_ppm=0.15, output_ppm=0.60):
    INPUT_PRICE_PER_MILLION = input_ppm
    OUTPUT_PRICE_PER_MILLION = output_ppm

    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_MILLION
    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model_name": model_name
    }