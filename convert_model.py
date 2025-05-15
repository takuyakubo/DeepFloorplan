import tensorflow as tf
import os

def convert_meta_to_saved_model(meta_path, checkpoint_path, export_dir):
    """
    古いTensorFlowモデル（.metaファイル）をSavedModel形式に変換します。
    
    Args:
        meta_path: .metaファイルのパス
        checkpoint_path: チェックポイントファイルのパス（拡張子なし）
        export_dir: 出力先ディレクトリ
    """
    # セッションの作成
    with tf.compat.v1.Session() as sess:
        # メタグラフの読み込み
        saver = tf.compat.v1.train.import_meta_graph(meta_path)
        # チェックポイントの復元
        saver.restore(sess, checkpoint_path)
        
        # グラフの取得
        graph = tf.compat.v1.get_default_graph()
        
        # 入力と出力のテンソルを取得
        inputs = graph.get_tensor_by_name('inputs:0')
        room_type = graph.get_tensor_by_name('Cast:0')
        room_boundary = graph.get_tensor_by_name('Cast_1:0')
        
        # SavedModel用のシグネチャを定義
        signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': tf.compat.v1.saved_model.utils.build_tensor_info(inputs)},
            outputs={
                'Cast:0': tf.compat.v1.saved_model.utils.build_tensor_info(room_type),
                'Cast_1:0': tf.compat.v1.saved_model.utils.build_tensor_info(room_boundary)
            },
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
        
        # SavedModelのエクスポート
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()

if __name__ == '__main__':
    # 変換パラメータの設定
    base_dir = './pretrained/pretrained_r3d'
    meta_path = os.path.join(base_dir, 'pretrained_r3d.meta')
    checkpoint_path = os.path.join(base_dir, 'pretrained_r3d')
    export_dir = './pretrained/pretrained_r3d_saved_model'
    
    # 出力ディレクトリの作成
    os.makedirs(export_dir, exist_ok=True)
    
    # モデルの変換
    convert_meta_to_saved_model(meta_path, checkpoint_path, export_dir)
    print(f'モデルを {export_dir} に変換しました。') 