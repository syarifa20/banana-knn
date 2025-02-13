def get_panen_status(prediction, confidence):
    """
    Menentukan apakah pisang layak panen berdasarkan prediksi dan confidence score.
    """
    if confidence < 0.6:
        return "Tidak dapat dipastikan"
    
    if prediction in ["matang", "terlalu-matang"]:
        return "Layak Panen"
    
    return "Belum Layak Panen"
