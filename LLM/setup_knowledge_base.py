import json
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

with open("menu.json", "r", encoding="utf-8") as file:
    menu_data = json.load(file)

documents = []
for danh_muc in menu_data["danh_muc"]:
    ten_danh_muc = danh_muc["ten_danh_muc"]
    danh_sach_mon = danh_muc["danh_sach_mon"]
    for item in danh_sach_mon:
        ten_mon = item["ten_mon"]
        mo_ta = item["mo_ta"]
        gia_co_ban = item["gia_co_ban"]
        thoi_gian_chuan_bi = item["thoi_gian_chuan_bi"]
        tuy_chon = []
        for i in item["danh_muc_tuy_chon"]:
            tuy_chon_list = ','.join([f"{e['ten']}(Giá thêm: {e['gia_them']})" for e in i["tuy_chon"]])
            tuy_chon_content = f"{i['ten_danh_muc']}({i['loai_lua_chon']}), Bao gồm: {tuy_chon_list}"
            tuy_chon.append(tuy_chon_content)
        tuy_chon_str = " | ".join(tuy_chon)
        content = f"Tên danh muc: {ten_danh_muc}, Tên món: {ten_mon}, Mô tả: {mo_ta}, Giá cơ bản: {gia_co_ban}, Thời gian chuẩn bị: {thoi_gian_chuan_bi}, Tùy chọn: {tuy_chon_str}"
        document = Document(
            page_content=content,
            metadata={
                "id_mon": item["id_mon"],
                "ten_mon": item["ten_mon"],
                "gia_co_ban": item["gia_co_ban"]
            }
        )
        documents.append(document)

model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vector_store = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="chroma_db"
)
